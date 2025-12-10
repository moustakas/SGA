"""
SGA.cutouts
===========

Utilities for generating large numbers of (annotated) cutouts.

"""
import pdb

import os, sys, re, time
import numpy as np
from glob import glob
import multiprocessing
from astropy.table import Table, vstack

from SGA.geometry import choose_geometry
from SGA.io import get_raslice
from SGA.SGA import sga2025_name
from SGA.logger import log


def get_pixscale_and_width(diam, mindiam=None, rescale=False, maxdiam_arcmin=25.,
                           default_width=152, default_pixscale=0.262):
    """Simple function to compute the pixel scale of the desired
    output images.

    """
    if mindiam is None:
        mindiam = default_width * default_pixscale # [arcsec]

    nobj = len(diam)

    if rescale:
        # scale the pixel scale so that larger objects "fit" in DEFAULT_WIDTH
        pixscale = default_pixscale * 1.5 * diam / mindiam   # [arcsec/pixel]
        width = np.zeros(nobj, int) + default_width # [pixels]
    else:
        # full-mosaic, native resolution width, except for objects
        # larger than XX arcmin
        pixscale = np.zeros(nobj) + default_pixscale # [arcsec/pixel]
        width = 1.5 * diam / pixscale # [pixels]

        maxdiam = maxdiam_arcmin * 60. # [arcsec]
        I = diam > maxdiam
        if np.any(I):
            pixscale[I] = default_pixscale * diam[I] / maxdiam
            width[I] = 1.5 * diam[I] / pixscale[I]

    width = width.astype(int)

    return pixscale, width


def cutouts_plan(cat, width=152, layer='ls-dr9', cutoutdir='.', annotatedir='.',
                 photodir='.', size=1, mp=1, group=False, photo=False,
                 gather_photo=False, annotate=False, fits_cutouts=True,
                 unwise_cutouts=False, galex_cutouts=False, overwrite=False,
                 verbose=False, use_catalog_objname=False):
    """Build a plan for generating (annotated) cutouts and basic photometry.

    """
    import multiprocessing

    t0 = time.time()

    if group:
        objname = cat['SGAGROUP']
        #objname = cat['GROUP_NAME']
    else:
        if use_catalog_objname:
            objname = cat['OBJNAME'].value
        else:
            objname = sga2025_name(cat['RA'], cat['DEC'], unixsafe=True)

    if photo or gather_photo:
        mpargs = [(obj, objname1, cutoutdir, photodir, gather_photo, overwrite, verbose)
                  for obj, objname1 in zip(cat, objname)]
        if mp > 1:
            with multiprocessing.Pool(mp) as P:
                out = P.map(_get_photo_filename, mpargs)
        else:
            out = [get_photo_filename(*mparg) for mparg in mpargs]
        out = list(zip(*out))

        fitsfiles = np.array(out[0], dtype=object)
        jpgfiles = np.array(out[1], dtype=object)
        photfiles = np.array(out[2], dtype=object)
        qafiles = np.array(out[3], dtype=object)
        nobj = np.array(out[4], dtype=object)
    elif annotate:
        mpargs = [(obj, objname1, cutoutdir, annotatedir, overwrite, verbose)
                  for obj, objname1 in zip(cat, objname)]
        if mp > 1:
            with multiprocessing.Pool(mp) as P:
                out = P.map(_get_annotate_filename, mpargs)
        else:
            out = [get_annotate_filename(*mparg) for mparg in mpargs]
        out = list(zip(*out))

        jpgfiles = np.array(out[0], dtype=object)
        pngfiles = np.array(out[1], dtype=object)
        nobj = np.array(out[2], dtype=object)
    else:
        if np.isscalar(width):
            width = [width] * len(objname)

        mpargs = [(obj, objname1, cutoutdir, width1, group, fits_cutouts,
                   unwise_cutouts, galex_cutouts, overwrite, verbose)
                  for obj, objname1, width1 in zip(cat, objname, width)]
        if mp > 1:
            with multiprocessing.Pool(mp) as P:
                out = P.map(_get_basefiles_one, mpargs)
        else:
            out = [get_basefiles_one(*mparg) for mparg in mpargs]
        out = list(zip(*out))

        basefiles = np.array(out[0], dtype=object)
        allra = np.array(out[1], dtype=object)
        alldec = np.array(out[2], dtype=object)
        nobj = np.array(out[3], dtype=object)

    iempty = np.where(nobj == 0)[0]
    if len(iempty) > 0:
        if gather_photo:
            log.info(f'Missing {len(iempty):,d} photometry file(s).')
        elif photo:
            log.info(f'Skipping {len(iempty):,d} object(s) with existing photometry files (or missing FITS cutouts).')
        elif annotate:
            log.info(f'Skipping {len(iempty):,d} object(s) with existing annotated images.')
        else:
            log.info(f'Skipping {len(iempty):,d} object(s) with existing cutouts.')

    itodo = np.where(nobj > 0)[0]
    if len(itodo) > 0:
        if gather_photo:
            log.info(f'Gathered photometry file names for {np.sum(nobj[itodo]):,d} objects.')
        elif photo:
            log.info(f'Photometry files needed for {np.sum(nobj[itodo]):,d} objects.')
        elif annotate:
            log.info(f'Annotated images needed for {np.sum(nobj[itodo]):,d} objects.')
        else:
            log.info(f'Cutouts needed for {np.sum(nobj[itodo]):,d} objects.')
        groups = np.array_split(itodo, size) # unweighted distribution
    else:
        groups = [np.array([])]

    if photo or gather_photo:
        return fitsfiles, jpgfiles, photfiles, qafiles, groups
    elif annotate:
        return jpgfiles, pngfiles, groups
    else:
        return basefiles, allra, alldec, groups


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


def _get_annotate_filename(args):
    return get_annotate_filename(*args)


def get_annotate_filename(obj, objname, cutoutdir, annotatedir,
                          overwrite=False, verbose=False):
    raslice = get_raslice(obj['RA'])

    if objname is None:
        brick = custom_brickname(obj['RA'], obj['DEC'])
        jpgfile = os.path.join(cutoutdir, raslice, brick[:6], f'{brick}.jpeg')
        pngfile = os.path.join(annotatedir, raslice, brick[:6], f'{brick}.png')
    else:
        jpgfile = os.path.join(cutoutdir, raslice, f'{objname}.jpeg')
        pngfile = os.path.join(annotatedir, raslice, f'{objname}.png')
    nobj = 1

    if overwrite is False:
        if os.path.isfile(pngfile):
            nobj = 0
            if verbose:
                log.info(f'Skipping existing annotated cutout {pngfile}')
    else:
        if not os.path.isfile(jpgfile):
            nobj = 0
            log.warning(f'Missing input cutout {jpgfile}')

    return jpgfile, pngfile, nobj


def _annotate_one(args):
    return annotate_one(*args)


def annotate_one(jpgfile, pngfile, objname, commonname, pixscale,
                 mosaic_diam, draw_largest_ellipse, primary, group):
    """Annotate one image.

    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from astropy.wcs import WCS
    from astropy.io import fits

    from SGA.sky import simple_wcs
    from SGA.geometry import parse_geometry
    from SGA.qa import overplot_ellipse


    if not os.path.isfile(jpgfile):
        return

    bbox = dict(boxstyle='round', facecolor='k', alpha=0.5)
    ref_pixscale = 0.262
    barlen = 15. / pixscale # [pixels]
    barlabel = '15 arcsec'

    N = len(group)
    primary_ra, primary_dec = primary['RA'], primary['DEC']
    row_parent = primary['ROW_PARENT']

    img = mpimg.imread(jpgfile)
    width = img.shape[0]
    wcs = simple_wcs(primary_ra, primary_dec, width, pixscale=pixscale)

    ellipse_colors = {'RC3': 'yellow', 'SMUDGes': 'orange', 'LVD': 'violet',
                      'SGA2020': 'dodgerblue', 'HYPERLEDA': 'red',
                      'ESO': 'pink', 'SDSS': 'pink', 'TWOMASS': 'pink',
                      'BASIC': 'pink', 'LIT': 'pink', 'CUSTOM': 'pink',
                      'NONE': 'pink', '': 'pink'}
    ellipse_linestyles = {'RC3': 'solid', 'SMUDGes': 'solid', 'LVD': 'solid',
                          'SGA2020': 'dashed', 'HYPERLEDA': 'dashdot',
                          'ESO': 'dashed', 'SDSS': 'dashed', 'TWOMASS': 'dashed',
                          'BASIC': 'dashed', 'LIT': 'dashed', 'CUSTOM': 'dashed',
                          'NONE': 'dashed', '': 'dashed'}

    outdir = os.path.dirname(pngfile)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    #pngfile = '/global/homes/i/ioannis/ioannis/tmp/'+os.path.basename(pngfile)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(img, origin='lower')
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)

    # only keep objects in the image
    keep = np.ones(len(group), bool)
    for imem, onegal in enumerate(group):
        xpix, ypix = wcs.wcs_world2pix(onegal['RA'], onegal['DEC'], 1)
        if xpix < 0 or ypix < 0 or xpix > width or ypix > width:
            keep[imem] = False
    group = group[keep]

    #print('HACK!!!')
    #if len(group) > 500:
    if len(group) > 10:
        log.warning('Too many group members; keeping just the 10 largest objects!')
        # group includes primary, so we need to be sure it doesn't get removed
        group_noprimary = group.copy()
        group_noprimary.remove_row(np.where(primary['ROW_PARENT'] == group['ROW_PARENT'])[0][0])
        # set mindiam=0 to prioritize existing diameter estimates
        diam, _, _, _ = choose_geometry(group_noprimary, mindiam=0.)
        srt = np.argsort(diam)[::-1]
        group_noprimary = group_noprimary[srt[:10]]
        group = vstack((Table(primary), group_noprimary))
        del group_noprimary

    objnames, xpixes, ypixes = [], [], []
    for imem, onegal in enumerate(group):
        ra = onegal['RA']
        dec = onegal['DEC']
        xpix, ypix = wcs.wcs_world2pix(ra, dec, 1)
        #if xpix < 0 or ypix < 0 or xpix > width or ypix > width:
        #    continue

        if onegal['OBJNAME'] != primary['OBJNAME']:
            objnames.append(onegal['OBJNAME'])
            xpixes.append(xpix)
            ypixes.append(ypix)

        if draw_largest_ellipse:
            diam, ba, pa, ref = choose_geometry(Table(onegal))
            diam = diam[0]
            ba = ba[0]
            pa = pa[0]
            ref = ref[0]
            if diam > 0.:
                if onegal['OBJNAME'] == primary['OBJNAME']:
                    majorminor = True
                else:
                    majorminor = False
                overplot_ellipse(diam, ba, pa, xpix, ypix, height_pixels=width,
                                 pixscale=pixscale, ax=ax, color=ellipse_colors[ref],
                                 linestyle=ellipse_linestyles[ref],
                                 draw_majorminor_axes=majorminor, jpeg=True)
        else:
            for ref in ['SGA2020', 'HYPERLEDA', 'LIT']:
                diam, ba, pa, outref = parse_geometry(Table(onegal), ref)
                #print(onegal['OBJNAME'], outref, diam, ellipse_colors[outref])
                if diam > 0.:
                    #print(onegal['OBJNAME'], ref, xpix, ypix, diam[0], ba[0], pa[0])
                    if onegal['OBJNAME'] == primary['OBJNAME']:
                        majorminor = True
                    else:
                        majorminor = False
                    overplot_ellipse(diam, ba, pa, xpix, ypix, height_pixels=width,
                                     pixscale=pixscale, ax=ax, color=ellipse_colors[outref],
                                     linestyle=ellipse_linestyles[outref],
                                     draw_majorminor_axes=majorminor, jpeg=True)

    # now annotate
    if len(objnames) > 0:
        def label_neighbor(objname, xy, xyname, xytext, ha='center', va='top'):
            ax.annotate('', xy=xy, xytext=xytext, annotation_clip=True,
                        arrowprops={'arrowstyle': '-', 'color': 'white'})
                        #dict(facecolor='white', edgecolor='white', width=0.5,
                        #     headwidth=2, shrink=0.005, alpha=0.75))
            ax.annotate(objname, xy=xyname, xytext=xytext, va=va, ha=ha,
                        color='white', bbox=bbox, fontsize=9,
                        annotation_clip=True)

        objnames = np.array(objnames)
        xpixes = np.array(xpixes)
        ypixes = np.array(ypixes)

        ysrt = np.argsort(ypixes)
        objnames = objnames[ysrt]
        xpixes = xpixes[ysrt]
        ypixes = ypixes[ysrt]

        lhs = xpixes < width / 2
        rhs = xpixes >= width / 2
        Nlhs = np.sum(lhs)
        Nrhs = np.sum(rhs)

        xmargin = 0.15 * width
        ymargin = 0.1 * width # 10% pixel margin
        if Nlhs > 0:
            lhs_yoffset = np.linspace(ymargin*2, width-ymargin*2, Nlhs)
            lhs_xoffset = xmargin + 0.1 * width * np.sin(np.linspace(0, 2*np.pi, Nlhs))
            for objname, xpix, ypix, xoffset, yoffset in zip(
                    objnames[lhs], xpixes[lhs], ypixes[lhs], lhs_xoffset, lhs_yoffset):
                xy = (xpix, width-ypix)
                xyname = (xoffset, width-yoffset)
                xytext = (xoffset, width-yoffset)
                #print(objname, xy, xyname, xytext)
                label_neighbor(objname, xy, xyname, xytext, ha='center', va='top')
        if Nrhs > 0:
            rhs_yoffset = np.linspace(ymargin*3, width-ymargin*2, Nrhs)
            rhs_xoffset = width - (xmargin + 0.1 * width * np.sin(np.linspace(0, 2*np.pi, Nrhs)))
            for objname, xpix, ypix, xoffset, yoffset in zip(
                    objnames[rhs], xpixes[rhs], ypixes[rhs], rhs_xoffset, rhs_yoffset):
                xy = (xpix, width-ypix)
                yname = width - yoffset
                # shift the position if the label is too close to the object
                if (yoffset - ypix) / width < 0.1:
                    yname /= 1.1
                elif (ypix - yoffset) / width < 0.1:
                    yname *= 1.1
                xyname = (xoffset, yname)
                xytext = (xoffset, yname)
                #print(objname, xy, xyname, xytext)
                label_neighbor(objname, xy, xyname, xytext, ha='center', va='top')

    ax.invert_yaxis() # JPEG is flipped relative to my FITS WCS
    ax.axis('off')
    if primary['MORPH'].strip() == '':
        morph = primary["OBJTYPE"].strip()
    else:
        morph = primary["OBJTYPE"].strip() +'; '+re.sub(r'\s+', ' ', primary["MORPH"])

    txt = '\n'.join([commonname, #objname.replace('_', ' '),
                     morph, f'{primary_ra:.7f}, {primary_dec:.6f}'])
                     #r'$(\alpha,\delta)$='+f'({primary_ra:.7f}, {primary_dec:.6f})'])
    #txt = '\n'.join([commonname+f' {morph}', objname.replace('_', ' '),
    #                 r'$(\alpha,\delta)$='+f'({primary_ra:.7f}, {primary_dec:.6f})'])
    ax.text(0.03, 0.93, txt, transform=ax.transAxes, ha='left', va='center',
            color='white', bbox=bbox, linespacing=1.5, fontsize=10)

    # add the scale bar
    xpos, ypos = 0.07, 0.07
    dx = barlen / img.shape[0]
    ax.plot([xpos, xpos+dx], [ypos, ypos], transform=ax.transAxes,
            color='white', lw=2)
    ax.text(xpos + dx/2., ypos+0.02, barlabel, transform=ax.transAxes,
            ha='center', va='center', color='white')
    ax.text(1-xpos, ypos, str(row_parent), transform=ax.transAxes,
            ha='right', va='center', color='white')

    fig.tight_layout()
    fig.savefig(pngfile, bbox_inches=0)#, dpi=200)
    plt.close()
    log.info(f'Wrote {pngfile}')


def do_annotate(cat, fullcat=None, default_width=152, default_pixscale=0.262,
                comm=None, mp=1, base_cutoutdir='.', cutoutdir='.', annotatedir='.',
                region='dr9-north', fits_cutouts=True, draw_largest_ellipse=False,
                httpdir=None, overwrite=False, debug=False, annotate_central_only=False,
                dry_run=False, verbose=False):
    """Wrapper to set up the full set of annotations.

    """
    from astrometry.libkd.spherematch import match_radec

    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    if rank == 0:
        t0 = time.time()
        mindiam = default_width * default_pixscale # [arcsec]
        diam, ba, pa, ref = choose_geometry(cat, mindiam=mindiam)

        pixscale, width = get_pixscale_and_width(
            diam, mindiam, rescale=False,
            default_width=default_width,
            default_pixscale=default_pixscale)

        jpgfiles, pngfiles, groups = cutouts_plan(
            cat, size=size, cutoutdir=cutoutdir, annotatedir=annotatedir,
            overwrite=overwrite, mp=mp, fits_cutouts=fits_cutouts,
            verbose=verbose, annotate=True)
        log.info(f'Planning took {time.time() - t0:.2f} sec')

        # write out an inventory file
        if httpdir:
            objnames = sga2025_name(cat['RA'].value, cat['DEC'].value, unixsafe=True)
            inventoryfile = os.path.join(base_cutoutdir, f'inventory-{region}.txt')
            with open(inventoryfile, 'w') as F:
                for objname, pngfile in zip(objnames, pngfiles):
                    F.write(f'{pngfile.replace(base_cutoutdir, httpdir)},{objname}\n')
            log.info(f'Wrote {inventoryfile}')
    else:
        jpgfiles, pngfiles, groups = [], [], []
        pixscale, diam = [], []

    if comm:
        jpgfiles = comm.bcast(jpgfiles, root=0)
        pngfiles = comm.bcast(pngfiles, root=0)
        groups = comm.bcast(groups, root=0)
        pixscale = comm.bcast(pixscale, root=0)
        diam = comm.bcast(diam, root=0)
    sys.stdout.flush()

    # all done
    if len(jpgfiles) == 0 or len(np.hstack(jpgfiles)) == 0:
        return

    assert(len(groups) == size)

    log.info(f'Rank {rank} started at {time.asctime()}')
    sys.stdout.flush()

    indx = groups[rank]
    if len(indx) == 0:
        return

    commonname = cat['OBJNAME'][indx].value
    objname = sga2025_name(cat['RA'][indx].value, cat['DEC'][indx].value, unixsafe=True)

    # initial match
    allmatches = match_radec(cat['RA'][indx].value, cat['DEC'][indx].value,
                             fullcat['RA'].value, fullcat['DEC'].value,
                             2.*np.max(diam)/3600., indexlist=True, notself=False)

    mpargs = []
    for iobj in range(len(indx)):
        #print(iobj)
        primary = cat[indx[iobj]]
        if annotate_central_only:
            group = Table(primary)
        else:
            # refine the search to this object's diameter
            m1, m2, _ = match_radec(primary['RA'], primary['DEC'], fullcat['RA'][allmatches[iobj]],
                                    fullcat['DEC'][allmatches[iobj]], 2.*diam[indx[iobj]]/3600.)
            group = fullcat[allmatches[iobj]][m2]
        if debug:
            print(primary['OBJNAME'])
            for one in group:
                if one['OBJNAME'] != primary['OBJNAME']:
                    print(one['OBJNAME'])
            print()
        else:
            mpargs.append((jpgfiles[indx[iobj]], pngfiles[indx[iobj]], objname[iobj], commonname[iobj],
                           pixscale[indx[iobj]], diam[indx[iobj]], draw_largest_ellipse,
                           primary, group))

    if debug:
        return

    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            P.map(_annotate_one, mpargs)
    else:
        [annotate_one(*mparg) for mparg in mpargs]

    sys.stdout.flush()

    #if comm is not None:
    #    comm.barrier()

    if rank == 0 and not dry_run:
        log.info(f'All done at {time.asctime()}')


def _cutout_one(args):
    return cutout_one(*args)


def cutout_one(basefile, ra, dec, optical_width, optical_pixscale,
               optical_layer, optical_bands, dry_run, fits_cutouts,
               ivar_cutouts, unwise_cutouts, galex_cutouts, rank, iobj):
    """
    pixscale = 0.262
    width = int(30 / pixscale)   # =114
    height = int(width / 1.3) # =87 [3:2 aspect ratio]

    shifterimg pull dstndstn/viewer-cutouts:latest
    shifter --image dstndstn/viewer-cutouts cutout --output cutout.jpg --ra 234.2915 --dec 16.7684 --size 256 --layer ls-dr9 --pixscale 0.262 --force

    """
    from cutout import cutout

    suffixes = ['.jpeg', ]
    layers = [optical_layer, ]
    pixscales = [optical_pixscale, ]
    widths = [optical_width, ]
    allbands = [optical_bands, ]

    if fits_cutouts:
        suffixes += ['.fits', ]
        layers += [optical_layer, ]
        pixscales += [optical_pixscale, ]
        widths += [optical_width, ]
        allbands += [optical_bands, ]
    if unwise_cutouts:
        unwise_width = int(optical_width * optical_pixscale / UNWISE_PIXSCALE)
        unwise_suffixes = ['-W1W2.fits', '-W3W4.fits', ]
        suffixes += unwise_suffixes
        layers += ['unwise-neo7', 'unwise-w3w4', ]
        pixscales += [UNWISE_PIXSCALE, UNWISE_PIXSCALE, ]
        widths += [unwise_width, unwise_width, ]
        allbands += [['1', '2'], ['3', '4'], ]
    if galex_cutouts:
        galex_width = int(optical_width * optical_pixscale / GALEX_PIXSCALE)
        suffixes += ['-galex.fits', ]
        layers += ['galex', ]
        pixscales += [GALEX_PIXSCALE, ]
        widths += [galex_width, ]
        allbands += [['f', 'n'], ]

    for suffix, layer, pixscale, width, bands in zip(suffixes, layers, pixscales, widths, allbands):
        outfile = basefile+suffix
        cmdargs = f'--output={outfile} --ra={ra} --dec={dec} --size={width} ' + \
            f'--layer={layer} --pixscale={pixscale} --bands={",".join(bands)} --force'
        if suffix != '.jpeg' and ivar_cutouts:
            cmdargs += ' --invvar'

        if dry_run:
            if suffix == '.jpeg':
                log.info(f'Rank {rank}, object {iobj}: cutout {cmdargs}')
        else:
            if suffix == '.jpeg':
                outdir = os.path.dirname(basefile)
                if not os.path.isdir(outdir):
                    os.makedirs(outdir, exist_ok=True)
            try:
                cutout(ra, dec, outfile, size=width, layer=layer, pixscale=pixscale,
                       invvar=ivar_cutouts, bands=bands, force=True)
                #if suffix == '.jpeg':
                #    print(f'Rank {rank}, object {iobj}: cutout {cmdargs}')
                log.info(f'Rank {rank}, object {iobj}: cutout {cmdargs}')
            except:
                if suffix == '.jpeg':
                    log.warning(f'Rank {rank}, object {iobj} off the footprint: cutout {cmdargs}')

    # merge the W1W2 and W3W4 files
    if unwise_cutouts:
        for ii, suffix in enumerate(unwise_suffixes):
            infile = basefile+suffix
            if ii == 0:
                hdr = fitsio.read_header(infile)
                for key in ['BANDS', 'BAND0', 'BAND1', 'COMMENT']:
                    hdr.delete(key)
                hdr['BANDS'] = '1234'
                for band in range(4):
                    hdr[f'BAND{band}'] = str(band+1)
                hdr['NAXIS3'] = 4
                #hdr['EXTNAME'] = 'IMAGE'
                img = np.zeros((4, hdr['NAXIS2'], hdr['NAXIS1']), 'f4')
                img[:2, :, :] = fitsio.read(infile, ext=0)
                if ivar_cutouts:
                    hdr_ivar = fitsio.read_header(infile, ext=1)
                    for key in ['BANDS', 'BAND0', 'BAND1']:
                        hdr_ivar.delete(key)
                    hdr_ivar['BANDS'] = '1234'
                    for band in range(4):
                        hdr_ivar[f'BAND{band}'] = str(band+1)
                    hdr_ivar['NAXIS3'] = 4
                    #hdr_ivar['EXTNAME'] = 'INVVAR'
                    ivar = np.zeros_like(img)
                    ivar[:2, :, :] = fitsio.read(infile, ext=1)
            else:
                img[2:, :, :] = fitsio.read(infile, ext=0)
                if ivar_cutouts:
                    ivar[2:, :, :] = fitsio.read(infile, ext=1)
            os.remove(infile)
        outfile = basefile+'-unwise.fits'
        fitsio.write(outfile, img, header=hdr, clobber=True)
        if ivar_cutouts:
            fitsio.write(outfile, ivar, header=hdr_ivar)


def _get_basefiles_one(args):
    return get_basefiles_one(*args)


def get_basefiles_one(obj, objname, cutoutdir, width=None, group=False,
                      fits_cutouts=True, unwise_cutouts=False, galex_cutouts=False,
                      overwrite=False, verbose=False):

    if group:
        racolumn = 'GROUP_RA'
        deccolumn = 'GROUP_DEC'
    else:
        racolumn = 'RA'
        deccolumn = 'DEC'

    raslice = get_raslice(obj[racolumn])

    if objname is None:
        brick = custom_brickname(obj[racolumn], obj[deccolumn])
        basefile = os.path.join(cutoutdir, raslice, brick[:6], brick)
    else:
        basefile = os.path.join(cutoutdir, raslice, objname)
    nobj = 1

    jpeg = os.path.isfile(basefile+'.jpeg')

    if overwrite is False:
        if fits_cutouts:
            fits = os.path.isfile(basefile+'.fits')
        else:
            fits = True
        if unwise_cutouts:
            unwise = os.path.isfile(basefile+'-unwise.fits')
        else:
            unwise = True
        if galex_cutouts:
            galex = os.path.isfile(basefile+'-galex.fits')
        else:
            galex = True

        if jpeg and fits and unwise and galex:
            # need to make sure the image is the correct size
            #width_exist = int(fitsio.read_header(basefile+'.fits')['IMAGEW'])
            #if width == width_exist:
            nobj = 0
            if verbose:
                log.info(f'Skipping existing cutout {basefile}.')

    return basefile, obj[racolumn], obj[deccolumn], nobj


def do_cutouts(cat, layer='ls-dr9', default_width=152, default_pixscale=0.262,
               default_bands=['g', 'r', 'i', 'z'], comm=None, mp=1, group=False,
               cutoutdir='.', base_cutoutdir='.', maxdiam_arcmin=25., rescale=False,
               diamcolumn=None, overwrite=False, fits_cutouts=True, ivar_cutouts=False,
               unwise_cutouts=False, galex_cutouts=False, dry_run=False,
               verbose=False, use_catalog_objname=False):

    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    if rank == 0:
        t0 = time.time()
        mindiam = default_width * default_pixscale # [arcsec]
        if diamcolumn:
            diam = cat[diamcolumn].value * 60. # [arcsec]
        elif group:
            # Is this a parent / sphere-grouped catalog?
            diam = cat['GROUP_DIAMETER'].value * 60. # [arcsec]
        else:
            diam, _, _, _ = choose_geometry(cat, mindiam=mindiam)

        pixscale, width = get_pixscale_and_width(
            diam, mindiam, rescale=rescale,
            maxdiam_arcmin=maxdiam_arcmin,
            default_width=default_width,
            default_pixscale=default_pixscale)

        basefiles, allra, alldec, groups = cutouts_plan(
            cat, width=width, layer=layer, cutoutdir=cutoutdir,
            size=size, group=group, overwrite=overwrite, mp=mp,
            fits_cutouts=fits_cutouts, unwise_cutouts=unwise_cutouts,
            galex_cutouts=galex_cutouts, verbose=verbose,
            use_catalog_objname=use_catalog_objname)
        log.info(f'Planning took {time.time() - t0:.2f} sec')
    else:
        basefiles, allra, alldec, groups = [], [], [], []
        pixscale, width = [], []

    if comm:
        basefiles = comm.bcast(basefiles, root=0)
        allra = comm.bcast(allra, root=0)
        alldec = comm.bcast(alldec, root=0)
        groups = comm.bcast(groups, root=0)
        pixscale = comm.bcast(pixscale, root=0)
        width = comm.bcast(width, root=0)
    sys.stdout.flush()

    # all done
    if len(basefiles) == 0 or len(np.hstack(basefiles)) == 0:
        return

    assert(len(groups) == size)

    log.info(f'Rank {rank} started at {time.asctime()}')
    sys.stdout.flush()

    indx = groups[rank]
    if len(indx) == 0:
        return

    mpargs = [(basefiles[indx[iobj]], allra[indx[iobj]], alldec[indx[iobj]],
               width[indx[iobj]], pixscale[indx[iobj]], layer, default_bands,
               dry_run, fits_cutouts, ivar_cutouts, unwise_cutouts, galex_cutouts,
               rank, iobj) for iobj in range(len(indx))]
    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            P.map(_cutout_one, mpargs)
    else:
        [cutout_one(*mparg) for mparg in mpargs]

    sys.stdout.flush()

    #if comm is not None:
    #    comm.barrier()

    if rank == 0 and not dry_run:
        log.info(f'All done at {time.asctime()}')


def annotated_montage(cat, cutoutdir='.', annotatedir='.', photodir='.',
                      region='dr9-north', npagemax=100, ssl_version=None,
                      rescale=False, photo=False, photo_version=None,
                      ssl=False, wisesize=False, lvd=False, zooniverse=False,
                      overwrite=False):
    """Build a single PDF file of annotated images, to enable fast
    visual inspection.

    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.image import imread
    from SGA.SGA import sga_dir

    if ssl and ssl_version is None:
        log.info('ssl_version must be specified')
        return

    if photo and photo_version is None:
        log.info('photo_version must be specified')
        return

    if photo:
        qadir = os.path.join(sga_dir(), 'parent', 'photo')
    elif ssl:
        qadir = os.path.join(sga_dir(), 'ssl', ssl_version)
    else:
        qadir = os.path.join(sga_dir(), 'parent', 'qa')

    if not os.path.isdir(qadir):
        os.makedirs(qadir, exist_ok=True)

    raslices = get_raslice(cat['RA'].value)
    objnames = sga2025_name(cat['RA'].value, cat['DEC'].value, unixsafe=True)
    if rescale:
        ext = '.jpeg'
        prefix = 'rescale'
        pngdir = cutoutdir
    elif photo:
        ext = '-phot.png'
        prefix = 'native'
        pngdir = photodir
    else:
        ext = '.png'
        prefix = 'annotated'
        pngdir = annotatedir

    I, pngfiles = [], []
    for ii, (raslice, objname) in enumerate(zip(raslices, objnames)):
        pngfile = os.path.join(pngdir, raslice, f'{objname}{ext}')
        if os.path.isfile(pngfile):
            pngfiles.append(pngfile)
            I.append(ii)
        else:
            log.info(f'Skipping missing file {pngfile}')
    pngfiles = np.array(pngfiles)
    I = np.array(I)
    cat = cat[I]

    if len(pngfiles) == 0:
        log.info(f'No color image files found in image directory {pngdir}')
        return
    #pngfiles = np.unique(pngfiles)

    if wisesize:
        #suffix = '-wisesize'
        pass
    elif lvd:
        #suffix = '-lvd'
        pass
    elif zooniverse:
        #suffix = '-zooniverse'
        pass
    elif ssl:
        suffix = f'-ssl-{ssl_version}'
    elif photo:
        suffix = f'-photo-{photo_version}'
    else:
        # attach the cutoutsdir basename to all output filenames
        suffix = f'-{os.path.basename(cutoutdir)}'

    #pngfiles = np.array(glob(os.path.join(outdir, region, 'annotate', '???', '*.png')))
    #pngfiles = pngfiles[np.argsort(pngfiles)]
    #pngfiles = pngfiles[:16]
    origindx = np.arange(len(pngfiles))
    nobj = len(origindx)

    if nobj <= 10:
        ncol, nrow = 1, 1
    elif (nobj > 10) * (nobj <= 50):
        ncol, nrow = 4, 4
    elif (nobj > 50) * (nobj >= 200):
        ncol, nrow = 6, 6
    else:
        ncol, nrow = 10, 10

    nperpage = ncol * nrow
    npage = int(np.ceil(len(pngfiles) / nperpage))

    # divide into multiple documents
    npdf = int(np.ceil(npage / npagemax))
    pdf_pngfiles = np.array_split(pngfiles, npdf)
    pdf_allindx = np.array_split(origindx, npdf)

    log.info(f'Distributing {len(pngfiles):,d} annotated images to {npdf:,d} ' + \
             f'PDFs with a total of {npage:,d} pages and {npagemax} pages per file.')

    #for ipdf in [1]:
    for ipdf in range(npdf):
        pdffile = os.path.join(qadir, f'{prefix}-montage-{region}{suffix}-{ipdf:03}.pdf')

        if os.path.isfile(pdffile) and not overwrite:
            log.info(f'Output file {pdffile} exists; use --overwrite')
            continue

        pngfiles = pdf_pngfiles[ipdf]
        orig_allindx = pdf_allindx[ipdf]
        allindx = np.arange(len(pngfiles))
        npage = int(np.ceil(len(pngfiles) / nperpage))

        pdf = PdfPages(pdffile)
        for ipage in range(npage):
            log.info(f'Building page {ipage+1:,d}/{npage:,d}')
            indx = allindx[ipage*nperpage:(ipage+1)*nperpage]
            origindx = orig_allindx[ipage*nperpage:(ipage+1)*nperpage]
            fig, ax = plt.subplots(nrow, ncol, figsize=(2*ncol, 2*nrow))
            for iax, xx in enumerate(np.atleast_1d(ax).flat):
                if iax < len(indx):
                    img = imread(pngfiles[indx[iax]])
                    xx.imshow(img, interpolation='None')
                    if rescale:
                        for spine in ['bottom', 'top', 'right', 'left']:
                            xx.spines[spine].set_color('white')
                        xx.set_xticks([])
                        xx.set_yticks([])
                        xx.text(0.9, 0.1, str(cat['ROW_PARENT'][origindx[iax]]), transform=xx.transAxes,
                                ha='right', va='center', color='white', fontsize=8)
                        xx.text(0.05, 0.9, str(cat['OBJNAME'][origindx[iax]]), transform=xx.transAxes,
                                ha='left', va='center', color='white', fontsize=6)
                        sz = img.shape
                        xx.add_artist(Circle((sz[1]/2., sz[0]/2.), radius=15./2/0.262,
                                             facecolor='none', edgecolor='yellow', ls='-', alpha=0.8))
                    else:
                        xx.axis('off')
                else:
                    xx.axis('off')
            fig.subplots_adjust(wspace=0., hspace=0., bottom=0.05, top=0.95, left=0.05, right=0.95)
            pdf.savefig(fig, dpi=150)
            plt.close()
        pdf.close()
        log.info(f'Wrote {pdffile}')

        if ssl and False:
            print(pdffile)
            for ipage in range(npage):
                origindx = orig_allindx[ipage*nperpage:(ipage+1)*nperpage]
                info = cat['OBJNAME', 'RA', 'DEC', 'ROW_PARENT'][origindx]
                info['PAGE'] = ipage
                _ = print(info.pprint(max_lines=-1))
            print()

