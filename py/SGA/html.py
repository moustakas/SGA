"""
SGA.html
========

Code to generate HTML content.

"""
import pdb

import os, subprocess
import numpy as np
from astropy.table import Table

from SGA.SGA import RACOLUMN, DECCOLUMN, DIAMCOLUMN, REFIDCOLUMN
from SGA.logger import log


def _get_cutouts_one(args):
    """Wrapper function for the multiprocessing."""
    return get_cutouts_one(*args)


def get_cutouts_one(group, clobber=False):
    """Get viewer cutouts for a single galaxy."""

    layer = get_layer(group)
    groupname = get_groupname(group)

    diam = group_diameter(group) # [arcmin]
    size = np.ceil(diam * 60 / PIXSCALE).astype('int') # [pixels]

    imageurl = '{}/?ra={:.8f}&dec={:.8f}&pixscale={:.3f}&size={:g}&layer={}'.format(
        cutouturl, group['ra'], group['dec'], PIXSCALE, size, layer)

    jpgfile = os.path.join(jpgdir, '{}.jpg'.format(groupname))
    cmd = 'wget --continue -O {:s} "{:s}"' .format(jpgfile, imageurl)
    if os.path.isfile(jpgfile) and not clobber:
        print('File {} exists...skipping.'.format(jpgfile))
    else:
        if os.path.isfile(jpgfile):
            os.remove(jpgfile)
        print(cmd)
        os.system(cmd)


def get_cutouts(groupsample, mp=1, clobber=False):
    """Get viewer cutouts of the whole sample."""

    cutoutargs = list()
    for gg in groupsample:
        cutoutargs.append( (gg, clobber) )

    if mp > 1:
        p = multiprocessing.Pool(mp)
        p.map(_get_cutouts_one, cutoutargs)
        p.close()
    else:
        for args in cutoutargs:
            _get_cutouts_one(args)
    return


def html_javadate():
    """Return a string that embeds a date in a webpage using Javascript.

    """
    import textwrap

    js = textwrap.dedent("""
    <SCRIPT LANGUAGE="JavaScript">
    var months = new Array(13);
    months[1] = "January";
    months[2] = "February";
    months[3] = "March";
    months[4] = "April";
    months[5] = "May";
    months[6] = "June";
    months[7] = "July";
    months[8] = "August";
    months[9] = "September";
    months[10] = "October";
    months[11] = "November";
    months[12] = "December";
    var dateObj = new Date(document.lastModified)
    var lmonth = months[dateObj.getMonth() + 1]
    var date = dateObj.getDate()
    var fyear = dateObj.getYear()
    if (fyear < 2000)
    fyear = fyear + 1900
    document.write(" " + fyear + " " + lmonth + " " + date)
    </SCRIPT>
    """)

    return js


def make_montage_coadds(galaxy, galaxydir, htmlgalaxydir, barlen=None,
                        barlabel=None, just_coadds=False, clobber=False,
                        verbose=False):
    """Montage the coadds into a nice QAplot.

    barlen - pixels

    """
    from SGA.qa import addbar_to_png, fonttype
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None

    for filesuffix in ['custom', 'pipeline']:
        montagefile = os.path.join(htmlgalaxydir, '{}-{}-montage-grz.png'.format(galaxy, filesuffix))
        thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-montage-grz.png'.format(galaxy, filesuffix))
        thumb2file = os.path.join(htmlgalaxydir, 'thumb2-{}-{}-montage-grz.png'.format(galaxy, filesuffix))
        if not os.path.isfile(montagefile) or clobber:
            if filesuffix == 'custom':
                coaddfiles = ('{}-image-grz'.format(filesuffix),
                              '{}-model-grz'.format(filesuffix),
                              '{}-resid-grz'.format(filesuffix))
            else:
                coaddfiles = ('{}-image-grz'.format(filesuffix),
                              '{}-model-grz'.format(filesuffix),
                              '{}-resid-grz'.format(filesuffix))

            # Image coadd with the scale bar label--
            barpngfile = os.path.join(htmlgalaxydir, '{}-{}.png'.format(galaxy, coaddfiles[0]))

            # Make sure all the files exist.
            check, _just_coadds = True, just_coadds
            jpgfile = []
            for suffix in coaddfiles:
                _jpgfile = os.path.join(galaxydir, '{}-{}.jpg'.format(galaxy, suffix))
                jpgfile.append(_jpgfile)
                if not os.path.isfile(_jpgfile):
                    if verbose:
                        print('File {} not found!'.format(_jpgfile))
                    check = False
                #print(check, _jpgfile)

            # Check for just the image coadd.
            if check is False:
                if os.path.isfile(np.atleast_1d(jpgfile)[0]):
                    _just_coadds = True
                else:
                    continue

            if check or _just_coadds:
                with Image.open(np.atleast_1d(jpgfile)[0]) as im:
                    sz = im.size
                if sz[0] > 4096 and sz[0] < 8192:
                    resize = '1024x1024'
                    #resize = '-resize 2048x2048 '
                elif sz[0] > 8192:
                    resize = '1024x1024'
                    #resize = '-resize 4096x4096 '
                else:
                    resize = None

                # Make a quick thumbnail of just the data.
                cmd = 'convert -thumbnail {0}x{0} {1} {2}'.format(96, np.atleast_1d(jpgfile)[0], thumb2file)
                if os.path.isfile(thumb2file):
                    os.remove(thumb2file)
                print('Writing {}'.format(thumb2file))
                subprocess.call(cmd.split())

                # Add a bar and label to the first image.
                if _just_coadds:
                    if resize:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -resize {} -geometry +0+0 '.format(resize)
                    else:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -geometry +0+0 '
                    if barlen:
                        addbar_to_png(jpgfile[0], barlen, barlabel, None, barpngfile, scaledfont=True)
                        cmd = cmd+' '+barpngfile
                    else:
                        cmd = cmd+' '+jpgfile
                    if sz[0] > 512:
                        thumbsz = 512
                    else:
                        thumbsz = sz[0]
                else:
                    if resize:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -resize {} -geometry +0+0 '.format(resize)
                    else:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -geometry +0+0 '
                    if barlen:
                        addbar_to_png(jpgfile[0], barlen, barlabel, None, barpngfile, scaledfont=True)
                        cmd = cmd+' '+barpngfile+' '
                        cmd = cmd+' '.join(ff for ff in jpgfile[1:])
                    else:
                        cmd = cmd+' '.join(ff for ff in jpgfile)
                    if sz[0] > 512:
                        thumbsz = 512*3
                    else:
                        thumbsz = sz[0]*3
                cmd = cmd+' {}'.format(montagefile)
                print(cmd)

                #if verbose:
                print('Writing {}'.format(montagefile))
                subprocess.call(cmd.split())
                if not os.path.isfile(montagefile):
                    print('There was a problem writing {}'.format(montagefile))
                    print(cmd)
                    continue

                # Create a couple smaller thumbnail images
                cmd = 'convert -thumbnail {0} {1} {2}'.format(thumbsz, montagefile, thumbfile)
                #print(cmd)
                if os.path.isfile(thumbfile):
                    os.remove(thumbfile)
                print('Writing {}'.format(thumbfile))
                subprocess.call(cmd.split())

                ## Create a couple smaller thumbnail images
                #for tf, sz in zip((thumbfile, thumb2file), (512, 96)):
                #    cmd = 'convert -thumbnail {}x{} {} {}'.format(sz, sz, montagefile, tf)
                #    #if verbose:
                #    print('Writing {}'.format(tf))
                #    subprocess.call(cmd.split())


def make_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir, refpixscale=0.262,
                                barlen=None, barlabel=None, just_coadds=False,
                                clobber=False, verbose=False):
    """Montage the GALEX and WISE coadds into a nice QAplot.

    barlen - pixels

    """
    from SGA.qa import addbar_to_png, fonttype
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None

    filesuffix = 'custom'

    for bandsuffix, pixscale in zip(('FUVNUV', 'W1W2'), (1.5, 2.75)):
        montagefile = os.path.join(htmlgalaxydir, '{}-{}-montage-{}.png'.format(galaxy, filesuffix, bandsuffix))
        thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-montage-{}.png'.format(galaxy, filesuffix, bandsuffix))
        thumb2file = os.path.join(htmlgalaxydir, 'thumb2-{}-{}-montage-{}.png'.format(galaxy, filesuffix, bandsuffix))
        if not os.path.isfile(montagefile) or clobber:
            coaddfiles = ('{}-image-{}'.format(filesuffix, bandsuffix),
                          '{}-model-{}'.format(filesuffix, bandsuffix),
                          '{}-resid-{}'.format(filesuffix, bandsuffix))

            # Image coadd with the scale bar label--
            barpngfile = os.path.join(htmlgalaxydir, '{}-{}.png'.format(galaxy, coaddfiles[0]))

            # Make sure all the files exist.
            check, _just_coadds = True, just_coadds
            jpgfile = []
            for suffix in coaddfiles:
                _jpgfile = os.path.join(galaxydir, '{}-{}.jpg'.format(galaxy, suffix))
                jpgfile.append(_jpgfile)
                if not os.path.isfile(_jpgfile):
                    if verbose:
                        print('File {} not found!'.format(_jpgfile))
                    check = False
                #print(check, _jpgfile)

            # Check for just the image coadd.
            if check is False:
                if os.path.isfile(np.atleast_1d(jpgfile)[0]):
                    _just_coadds = True
                else:
                    continue

            if check or _just_coadds:
                with Image.open(np.atleast_1d(jpgfile)[0]) as im:
                    sz = im.size
                if sz[0] > 4096 and sz[0] < 8192:
                    resize = '1024x1024'
                    #resize = '-resize 2048x2048 '
                elif sz[0] > 8192:
                    resize = '1024x1024'
                    #resize = '-resize 4096x4096 '
                else:
                    resize = None

                # Make a quick thumbnail of just the data.
                cmd = 'convert -thumbnail {0}x{0} {1} {2}'.format(96, np.atleast_1d(jpgfile)[0], thumb2file)
                if os.path.isfile(thumb2file):
                    os.remove(thumb2file)
                print('Writing {}'.format(thumb2file))
                subprocess.call(cmd.split())

                # Add a bar and label to the first image.
                if _just_coadds:
                    if resize:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -resize {} -geometry +0+0 '.format(resize)
                    else:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -geometry +0+0 '
                    if barlen:
                        addbar_to_png(jpgfile[0], barlen, barlabel, None, barpngfile, scaledfont=True)
                        cmd = cmd+' '+barpngfile
                    else:
                        cmd = cmd+' '+jpgfile
                    if sz[0] > 512:
                        thumbsz = 512
                    else:
                        thumbsz = sz[0]
                else:
                    if resize:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -resize {} -geometry +0+0 '.format(resize)
                    else:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -geometry +0+0 '
                    if barlen:
                        pixscalefactor = pixscale / refpixscale
                        #barlen2 = barlen / pixscalefactor
                        #pixscalefactor = 1.0
                        addbar_to_png(jpgfile[0], barlen, barlabel, None, barpngfile,
                                      scaledfont=True, pixscalefactor=pixscalefactor)
                        cmd = cmd+' '+barpngfile+' '
                        cmd = cmd+' '.join(ff for ff in jpgfile[1:])
                    else:
                        cmd = cmd+' '.join(ff for ff in jpgfile)
                    if sz[0] > 512:
                        thumbsz = 512*3
                    else:
                        thumbsz = sz[0]*3
                cmd = cmd+' {}'.format(montagefile)
                print(cmd)

                #if verbose:
                print('Writing {}'.format(montagefile))
                subprocess.call(cmd.split())
                if not os.path.isfile(montagefile):
                    print('There was a problem writing {}'.format(montagefile))
                    print(cmd)
                    continue

                # Create a couple smaller thumbnail images
                cmd = 'convert -thumbnail {0} {1} {2}'.format(thumbsz, montagefile, thumbfile)
                #print(cmd)
                if os.path.isfile(thumbfile):
                    os.remove(thumbfile)                
                print('Writing {}'.format(thumbfile))
                subprocess.call(cmd.split())
    
    
def make_maskbits_qa(galaxy, galaxydir, htmlgalaxydir, clobber=False, verbose=False):
    """Visualize the maskbits image.

    """
    import fitsio
    from SGA.qa import qa_maskbits

    filesuffix = 'largegalaxy'

    maskbitsfile = os.path.join(htmlgalaxydir, '{}-{}-maskbits.png'.format(galaxy, filesuffix))
    if not os.path.isfile(maskbitsfile) or clobber:
        fitsfile = os.path.join(galaxydir, '{}-{}-maskbits.fits.fz'.format(galaxy, filesuffix))
        tractorfile = os.path.join(galaxydir, '{}-{}-tractor.fits'.format(galaxy, filesuffix))
        if not os.path.isfile(fitsfile):
            if verbose:
                print('File {} not found!'.format(fitsfile))
            return
        if not os.path.isfile(tractorfile):
            if verbose:
                print('File {} not found!'.format(tractorfile))
            return

        mask = fitsio.read(fitsfile)
        tractor = fitsio.read(tractorfile)

        qa_maskbits(mask, tractor, png=maskbitsfile)


def make_ellipse_qa(galaxy, galaxydir, htmlgalaxydir, bands=['g', 'r', 'i', 'z'],
                    refband='r', pixscale=0.262, read_multiband=None,
                    qa_multiwavelength_sed=None, SBTHRESH=None,
                    linear=False, plot_colors=True,
                    galaxy_id=None, barlen=None, barlabel=None, clobber=False, verbose=False,
                    cosmo=None, galex=False, unwise=False, scaledfont=False):
    """Generate QAplots from the ellipse-fitting.

    """
    import fitsio
    from PIL import Image
    from SGA.qa import (display_multiband, display_ellipsefit,
                        display_ellipse_sbprofile, qa_curveofgrowth,
                        qa_maskbits)
    if qa_multiwavelength_sed is None:
        from SGA.qa import qa_multiwavelength_sed

    Image.MAX_IMAGE_PIXELS = None

    # Read the data.
    if read_multiband is None:
        print('Unable to build ellipse QA without specifying read_multiband method.')
        return

    data, galaxyinfo = read_multiband(galaxy, galaxydir, galaxy_id=galaxy_id, bands=bands,
                                      refband=refband, pixscale=pixscale,
                                      verbose=verbose, galex=galex, unwise=unwise)

    if not bool(data) or data['missingdata']:
        return

    if data['failed']: # all galaxies dropped
        return

    # optionally read the Tractor catalog
    tractor = None
    if galex or unwise:
        tractorfile = os.path.join(galaxydir, '{}-{}-tractor.fits'.format(galaxy, data['filesuffix']))
        if os.path.isfile(tractorfile):
            tractor = Table(fitsio.read(tractorfile, lower=True))

    ellipsefitall = []
    for igal, galid in enumerate(data['galaxy_id']):
        galid = str(galid)
        ellipsefit = SGA.io.read_ellipsefit(galaxy, galaxydir, filesuffix=data['filesuffix'],
                                            galaxy_id=galid, verbose=verbose)
        if bool(ellipsefit):
            ellipsefitall.append(ellipsefit)

            if galid.strip() != '':
                galid = '{}-'.format(galid)

            if galex or unwise:
                sedfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}sed.png'.format(galaxy, data['filesuffix'], galid))
                if not os.path.isfile(sedfile) or clobber:
                    _tractor = None
                    if tractor is not None:
                        _tractor = tractor[(tractor['ref_cat'] != '  ')*np.isin(tractor['ref_id'], data['galaxy_id'][igal])] # fragile...
                    qa_multiwavelength_sed(ellipsefit, tractor=_tractor, png=sedfile, verbose=verbose)

            sbprofilefile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}sbprofile.png'.format(galaxy, data['filesuffix'], galid))
            if not os.path.isfile(sbprofilefile) or clobber:
                display_ellipse_sbprofile(ellipsefit, plot_radius=False, plot_sbradii=False,
                                          png=sbprofilefile, verbose=verbose, minerr=0.0,
                                          cosmo=cosmo, linear=linear, plot_colors=plot_colors)

            cogfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}cog.png'.format(galaxy, data['filesuffix'], galid))
            if not os.path.isfile(cogfile) or clobber:
                qa_curveofgrowth(ellipsefit, pipeline_ellipsefit={}, plot_sbradii=False,
                                 png=cogfile, verbose=verbose, cosmo=cosmo)

            #print('hack!')
            #continue

            if unwise:
                multibandfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}multiband-W1W2.png'.format(galaxy, data['filesuffix'], galid))
                thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-ellipse-{}multiband-W1W2.png'.format(galaxy, data['filesuffix'], galid))
                if not os.path.isfile(multibandfile) or clobber:
                    with Image.open(os.path.join(galaxydir, '{}-{}-image-W1W2.jpg'.format(galaxy, data['filesuffix']))) as colorimg:
                        display_multiband(data, ellipsefit=ellipsefit, colorimg=colorimg,
                                          igal=igal, barlen=barlen, barlabel=barlabel,
                                          png=multibandfile, verbose=verbose, scaledfont=scaledfont,
                                          SBTHRESH=SBTHRESH,
                                          galex=False, unwise=True)
                    # Create a thumbnail.
                    cmd = 'convert -thumbnail 1024x1024 {} {}'.format(multibandfile, thumbfile)#.replace('>', '\>')
                    if os.path.isfile(thumbfile):
                        os.remove(thumbfile)
                    print('Writing {}'.format(thumbfile))
                    subprocess.call(cmd.split())

            if galex:
                multibandfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}multiband-FUVNUV.png'.format(galaxy, data['filesuffix'], galid))
                thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-ellipse-{}multiband-FUVNUV.png'.format(galaxy, data['filesuffix'], galid))
                if not os.path.isfile(multibandfile) or clobber:
                    with Image.open(os.path.join(galaxydir, '{}-{}-image-FUVNUV.jpg'.format(galaxy, data['filesuffix']))) as colorimg:
                        display_multiband(data, ellipsefit=ellipsefit, colorimg=colorimg,
                                          igal=igal, barlen=barlen, barlabel=barlabel,
                                          png=multibandfile, verbose=verbose, scaledfont=scaledfont,
                                          SBTHRESH=SBTHRESH,
                                          galex=True, unwise=False)
                    # Create a thumbnail.
                    cmd = 'convert -thumbnail 1024x1024 {} {}'.format(multibandfile, thumbfile)#.replace('>', '\>')
                    if os.path.isfile(thumbfile):
                        os.remove(thumbfile)
                    print('Writing {}'.format(thumbfile))
                    subprocess.call(cmd.split())

            multibandfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}multiband.png'.format(galaxy, data['filesuffix'], galid))
            thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-ellipse-{}multiband.png'.format(galaxy, data['filesuffix'], galid))
            if not os.path.isfile(multibandfile) or clobber:
                with Image.open(os.path.join(galaxydir, '{}-{}-image-grz.jpg'.format(galaxy, data['filesuffix']))) as colorimg:
                    display_multiband(data, ellipsefit=ellipsefit, colorimg=colorimg, bands=bands,
                                      igal=igal, barlen=barlen, barlabel=barlabel,
                                      SBTHRESH=SBTHRESH,
                                      png=multibandfile, verbose=verbose, scaledfont=scaledfont)

                # Create a thumbnail.
                cmd = 'convert -thumbnail 1024x1024 {} {}'.format(multibandfile, thumbfile)#.replace('>', '\>')
                if os.path.isfile(thumbfile):
                    os.remove(thumbfile)
                print('Writing {}'.format(thumbfile))
                subprocess.call(cmd.split())

            ## hack!
            #print('HACK!!!')
            #continue

    ## maskbits QA
    #maskbitsfile = os.path.join(htmlgalaxydir, '{}-{}-maskbits.png'.format(galaxy, data['filesuffix']))
    #if not os.path.isfile(maskbitsfile) or clobber:
    #    fitsfile = os.path.join(galaxydir, '{}-{}-maskbits.fits.fz'.format(galaxy, data['filesuffix']))
    #    tractorfile = os.path.join(galaxydir, '{}-{}-tractor.fits'.format(galaxy, data['filesuffix']))
    #    if not os.path.isfile(fitsfile):
    #        if verbose:
    #            print('File {} not found!'.format(fitsfile))
    #        return
    #    if not os.path.isfile(tractorfile):
    #        if verbose:
    #            print('File {} not found!'.format(tractorfile))
    #        return
    #
    #    mask = fitsio.read(fitsfile)
    #    tractor = Table(fitsio.read(tractorfile, upper=True))
    #
    #    with Image.open(os.path.join(galaxydir, '{}-{}-image-grz.jpg'.format(galaxy, data['filesuffix']))) as colorimg:
    #        qa_maskbits(mask, tractor, ellipsefitall, colorimg, largegalaxy=True, png=maskbitsfile)


def make_plots(galaxy, galaxydir, htmlgalaxydir, REFIDCOLUMN, read_multiband_function,
               unpack_maskbits_function, MASKBITS, run='south', mp=1,
               bands=['g', 'r', 'i', 'z'], radius_mosaic_arcsec=None,
               pixscale=0.262, barlen=None, barlabel=None, galex=True,
               unwise=True, verbose=False, clobber=False):

#sample, datadir=None, htmldir=None, survey=None, refband='r',
#               bands=['g', 'r', 'i', 'z'], pixscale=0.262, zcolumn='Z', galaxy_id=None,
#               mp=1, barlen=None, barlabel=None, SBTHRESH=None,
#               radius_mosaic_arcsec=None, linear=False, plot_colors=True,
#               clobber=False, verbose=True, get_galaxy_galaxydir=None,
#               read_multiband=None, qa_multiwavelength_sed=None,
#               cosmo=None, galex=False, unwise=False,
#               just_coadds=False, scaledfont=False):
    """Make QA plots.

    """
    import fitsio

    # Read the sample catalog for this group.
    samplefile = os.path.join(galaxydir, f'{galaxy}-sample.fits')

    sample = Table(fitsio.read(samplefile))#, columns=['SGAID', 'SGANAME']))
    log.info(f'Read {len(sample)} source(s) from {samplefile}')

    # Read the per-object ellipse-fitting results.
    for iobj, obj in enumerate(sample):
        suffix = ''.join(bands)
        ellipsefile = os.path.join(galaxydir, f'{galaxy}-ellipse-{obj[REFIDCOLUMN]}-{suffix}.fits')
        log.info(f'Reading {ellipsefile}')

        images = fitsio.read(ellipsefile, 'IMAGES')
        models = fitsio.read(ellipsefile, 'MODELS')
        maskbits = fitsio.read(ellipsefile, 'MASKBITS')
        ellipse = Table(fitsio.read(ellipsefile, 'ELLIPSE'))
        sbprofiles = Table(fitsio.read(ellipsefile, 'SBPROFILES'))

        pdb.set_trace()

        # Build the ellipse and photometry plots.
        if not just_coadds:
            make_ellipse_qa(galaxy, galaxydir, htmlgalaxydir, bands=bands, refband=refband,
                            pixscale=pixscale, barlen=barlen, barlabel=barlabel,
                            galaxy_id=galaxy_id, SBTHRESH=SBTHRESH,
                            linear=linear, plot_colors=plot_colors,
                            clobber=clobber, verbose=verbose, galex=galex, unwise=unwise,
                            cosmo=cosmo, scaledfont=scaledfont, read_multiband=read_multiband,
                            qa_multiwavelength_sed=qa_multiwavelength_sed)
            #continue # here!
            #pdb.set_trace()

        # Multiwavelength coadds (does not support just_coadds=True)--
        if galex:
            make_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir,
                                        refpixscale=pixscale,
                                        #barlen=barlen, barlabel=barlabel,
                                        clobber=clobber, verbose=verbose)

        # Build the montage coadds.
        make_montage_coadds(galaxy, galaxydir, htmlgalaxydir, barlen=barlen,
                            barlabel=barlabel, clobber=clobber, verbose=verbose,
                            just_coadds=just_coadds)

        # Build the maskbits figure.
        #make_maskbits_qa(galaxy, galaxydir, htmlgalaxydir, clobber=clobber, verbose=verbose)


    return 1


def skyserver_link(sdss_objid):
    return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(sdss_objid)


# Get the viewer link
def viewer_link(ra, dec, width, sga=False, manga=False, dr10=False):
    baseurl = 'http://legacysurvey.org/viewer-dev/'
    if width > 1200:
        zoom = 13
    elif (width > 400) * (width < 1200):
        zoom = 14
    else:
        zoom = 15

    if dr10:
        drlayer = 'ls-dr10'
    else:
        drlayer = 'ls-dr9'

    layer1 = ''
    if sga:
        layer1 = '&sga&sga-parent'
    if manga:
        layer1 = layer1+'&manga'

    viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer={}{}'.format(
        baseurl, ra, dec, zoom, drlayer, layer1)

    return viewer


def build_htmlhome(sample, htmldir, htmlhome='index.html', pixscale=0.262,
                   racolumn=RACOLUMN, deccolumn=DECCOLUMN, diamcolumn=DIAMCOLUMN,
                   html_raslices=True):
    """Build the home (index.html) page and, optionally, the trends.html top-level
    page.

    """
    htmlhomefile = os.path.join(htmldir, htmlhome)
    print('Building {}'.format(htmlhomefile))

    js = html_javadate()

    # group by RA slices
    raslices = np.array([SGA.io.get_raslice(ra) for ra in sample[racolumn]])
    #rasorted = raslices)

    with open(htmlhomefile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
        html.write('p {display: inline-block;;}\n')
        html.write('</style>\n')

        html.write('<h1>Virgo Filaments</h1>\n')

        html.write('<p style="width: 75%">\n')
        html.write("""This project is super neat.</p>\n""")

        # The default is to organize the sample by RA slice, but support both options here.
        if html_raslices:
            html.write('<p>The web-page visualizations are organized by one-degree slices of right ascension.</p><br />\n')

            html.write('<table>\n')
            html.write('<tr><th>RA Slice</th><th>Number of Galaxies</th></tr>\n')
            for raslice in sorted(set(raslices)):
                inslice = np.where(raslice == raslices)[0]
                html.write('<tr><td><a href="RA{0}.html"><h3>{0}</h3></a></td><td>{1}</td></tr>\n'.format(raslice, len(inslice)))
            html.write('</table>\n')
        else:
            html.write('<br /><br />\n')
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th> </th>\n')
            #html.write('<th>Index</th>\n')
            html.write('<th>ID</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Diameter (arcmin)</th>\n')
            html.write('<th>Viewer</th>\n')
            html.write('</tr>\n')

            galaxy, galaxydir, htmlgalaxydir = SGA.io.get_galaxy_galaxydir(sample, html=True)
            for gal, galaxy1, htmlgalaxydir1 in zip(sample, np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-custom-montage-grz.png'.format(galaxy1))
                thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-custom-montage-grz.png'.format(galaxy1))

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                link = viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

                html.write('<tr>\n')
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                #html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal[REFIDCOLUMN]))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(ra1))
                html.write('<td>{:.7f}</td>\n'.format(dec1))
                html.write('<td>{:.4f}</td>\n'.format(diam1))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(link))
                html.write('</tr>\n')
            html.write('</table>\n')

        # close up shop
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')

    # Optionally build the individual pages (one per RA slice).
    if html_raslices:
        for raslice in sorted(set(raslices)):
            inslice = np.where(raslice == raslices)[0]
            galaxy, galaxydir, htmlgalaxydir = SGA.io.get_galaxy_galaxydir(sample[inslice], region=region, html=True)

            slicefile = os.path.join(htmldir, 'RA{}.html'.format(raslice))
            print('Building {}'.format(slicefile))

            with open(slicefile, 'w') as html:
                html.write('<html><body>\n')
                html.write('<style type="text/css">\n')
                html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
                html.write('p {width: "75%";}\n')
                html.write('</style>\n')

                html.write('<h3>RA Slice {}</h3>\n'.format(raslice))

                html.write('<table>\n')
                html.write('<tr>\n')
                #html.write('<th>Number</th>\n')
                html.write('<th> </th>\n')
                #html.write('<th>Index</th>\n')
                html.write('<th>ID</th>\n')
                html.write('<th>Galaxy</th>\n')
                html.write('<th>RA</th>\n')
                html.write('<th>Dec</th>\n')
                html.write('<th>Diameter (arcmin)</th>\n')
                html.write('<th>Viewer</th>\n')

                html.write('</tr>\n')
                for gal, galaxy1, htmlgalaxydir1 in zip(sample[inslice], np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                    htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                    pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-custom-montage-grz.png'.format(galaxy1))
                    thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-custom-montage-grz.png'.format(galaxy1))

                    ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                    link = viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

                    html.write('<tr>\n')
                    #html.write('<td>{:g}</td>\n'.format(count))
                    #print(gal['INDEX'], gal[REFIDCOLUMN], gal['GALAXY'])
                    html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                    #html.write('<td>{}</td>\n'.format(gal['INDEX']))
                    html.write('<td>{}</td>\n'.format(gal[REFIDCOLUMN]))
                    html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                    html.write('<td>{:.7f}</td>\n'.format(ra1))
                    html.write('<td>{:.7f}</td>\n'.format(dec1))
                    html.write('<td>{:.4f}</td>\n'.format(diam1))
                    #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                    #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                    #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                    html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(link))
                    #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
                    html.write('</tr>\n')
                html.write('</table>\n')
                #count += 1

                html.write('<br /><br />\n')
                html.write('<b><i>Last updated {}</b></i>\n'.format(js))
                html.write('</html></body>\n')



def _build_htmlpage_one(args):
    """Wrapper function for the multiprocessing."""
    return build_htmlpage_one(*args)


def build_htmlpage_one(ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                       racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy, prevgalaxy,
                       nexthtmlgalaxydir, prevhtmlgalaxydir, verbose, clobber):
    """Build the web page for a single galaxy.

    """
    import fitsio
    from glob import glob
    import SGA.io

    if not os.path.exists(htmlgalaxydir1):
        os.makedirs(htmlgalaxydir1)

    htmlfile = os.path.join(htmlgalaxydir1, '{}.html'.format(galaxy1))
    if os.path.isfile(htmlfile) and not clobber:
        print('File {} exists and clobber=False'.format(htmlfile))
        return

    nexthtmlgalaxydir1 = os.path.join('{}'.format(nexthtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(nextgalaxy[ii]))
    prevhtmlgalaxydir1 = os.path.join('{}'.format(prevhtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(prevgalaxy[ii]))

    js = html_javadate()

    # Support routines--

    def _read_ccds_tractor_sample(prefix):
        nccds, tractor, sample = None, None, None

        ccdsfile = glob(os.path.join(galaxydir1, '{}-{}-ccds-*.fits'.format(galaxy1, prefix))) # north or south
        if len(ccdsfile) > 0:
            nccds = fitsio.FITS(ccdsfile[0])[1].get_nrows()

        # samplefile can exist without tractorfile when using --just-coadds
        samplefile = os.path.join(galaxydir1, '{}-sample.fits'.format(galaxy1))
        #samplefile = os.path.join(galaxydir1, '{}-{}-sample.fits'.format(galaxy1, prefix))
        if os.path.isfile(samplefile):
            sample = astropy.table.Table(fitsio.read(samplefile, upper=True))
            if verbose:
                print('Read {} galaxy(ies) from {}'.format(len(sample), samplefile))

        tractorfile = os.path.join(galaxydir1, '{}-{}-tractor.fits'.format(galaxy1, prefix))
        if os.path.isfile(tractorfile):
            cols = ['ref_cat', 'ref_id', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                    'flux_g', 'flux_r', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
                    'flux_fuv', 'flux_nuv', 'flux_ivar_fuv', 'flux_ivar_nuv', 
                    'flux_w1', 'flux_w2', 'flux_w3', 'flux_w4',
                    'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4']
            tractor = astropy.table.Table(fitsio.read(tractorfile, lower=True, columns=cols))#, rows=irows

            # We just care about the galaxies in our sample
            if prefix == 'custom':
                wt, ws = [], []
                for ii, sid in enumerate(sample[REFIDCOLUMN]):
                    ww = np.where((tractor['ref_cat'] != '  ') * (tractor['ref_id'] == sid))[0]
                    if len(ww) > 0:
                        wt.append(ww)
                        ws.append(ii)
                if len(wt) == 0:
                    print('All galaxy(ies) in {} field dropped from Tractor!'.format(galaxydir1))
                    tractor = None
                else:
                    wt = np.hstack(wt)
                    ws = np.hstack(ws)
                    tractor = tractor[wt]
                    sample = sample[ws]
                    srt = np.argsort(tractor['flux_r'])[::-1]
                    tractor = tractor[srt]
                    sample = sample[srt]
                    assert(np.all(tractor['ref_id'] == sample[REFIDCOLUMN]))

        return nccds, tractor, sample

    def _html_group_properties(html, gal):
        """Build the table of group properties.

        """
        ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
        link = viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

        html.write('<h2>Group Properties</h2>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        #html.write('<th>Number</th>\n')
        #html.write('<th>Index<br />(Primary)</th>\n')
        html.write('<th>ID<br />(Primary)</th>\n')
        html.write('<th>Group Name</th>\n')
        html.write('<th>Group RA</th>\n')
        html.write('<th>Group Dec</th>\n')
        html.write('<th>Group Diameter<br />(arcmin)</th>\n')
        #html.write('<th>Richness</th>\n')
        #html.write('<th>Pcen</th>\n')
        html.write('<th>Viewer</th>\n')
        #html.write('<th>SkyServer</th>\n')
        html.write('</tr>\n')

        html.write('<tr>\n')
        #html.write('<td>{:g}</td>\n'.format(ii))
        #print(gal['INDEX'], gal[REFIDCOLUMN], gal['GALAXY'])
        #html.write('<td>{}</td>\n'.format(gal['INDEX']))
        html.write('<td>{}</td>\n'.format(gal[REFIDCOLUMN]))
        html.write('<td>{}</td>\n'.format(gal['GROUP_NAME']))
        html.write('<td>{:.7f}</td>\n'.format(ra1))
        html.write('<td>{:.7f}</td>\n'.format(dec1))
        html.write('<td>{:.4f}</td>\n'.format(diam1))
        #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
        #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
        #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
        html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(link))
        #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
        html.write('</tr>\n')
        html.write('</table>\n')

        # Add the properties of each galaxy.
        html.write('<h3>Group Members</h3>\n')
        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>ID</th>\n')
        html.write('<th>Galaxy</th>\n')
        #html.write('<th>Morphology</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>D(25)<br />(arcmin)</th>\n')
        #html.write('<th>PA<br />(deg)</th>\n')
        #html.write('<th>e</th>\n')
        html.write('</tr>\n')
        for groupgal in sample:
            #if '031705' in gal['GALAXY']:
            #    print(groupgal['GALAXY'])
            html.write('<tr>\n')
            html.write('<td>{}</td>\n'.format(groupgal[REFIDCOLUMN]))
            html.write('<td>{}</td>\n'.format(groupgal[REFIDCOLUMN]))
            #typ = groupgal['MORPHTYPE'].strip()
            #if typ == '' or typ == 'nan':
            #    typ = '...'
            #html.write('<td>{}</td>\n'.format(typ))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['DEC']))
            html.write('<td>{:.4f}</td>\n'.format(groupgal['DIAM']))
            #if np.isnan(groupgal['PA']):
            #    pa = 0.0
            #else:
            #    pa = groupgal['PA']
            #html.write('<td>{:.2f}</td>\n'.format(pa))
            #html.write('<td>{:.3f}</td>\n'.format(1-groupgal['BA']))
            html.write('</tr>\n')
        html.write('</table>\n')

    def _html_image_mosaics(html):
        html.write('<h2>Image Mosaics</h2>\n')

        if False:
            html.write('<table>\n')
            html.write('<tr><th colspan="3">Mosaic radius</th><th colspan="3">Point-source depth<br />(5-sigma, mag)</th><th colspan="3">Image quality<br />(FWHM, arcsec)</th></tr>\n')
            html.write('<tr><th>kpc</th><th>arcsec</th><th>grz pixels</th><th>g</th><th>r</th><th>z</th><th>g</th><th>r</th><th>z</th></tr>\n')
            html.write('<tr><td>{:.0f}</td><td>{:.3f}</td><td>{:.1f}</td>'.format(
                radius_mosaic_kpc, radius_mosaic_arcsec, radius_mosaic_pixels))
            if bool(ellipse):
                html.write('<td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td>'.format(
                    ellipse['psfdepth_g'], ellipse['psfdepth_min_g'], ellipse['psfdepth_max_g'],
                    ellipse['psfdepth_r'], ellipse['psfdepth_min_r'], ellipse['psfdepth_max_r'],
                    ellipse['psfdepth_z'], ellipse['psfdepth_min_z'], ellipse['psfdepth_max_z']))
                html.write('<td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td></tr>\n'.format(
                    ellipse['psfsize_g'], ellipse['psfsize_min_g'], ellipse['psfsize_max_g'],
                    ellipse['psfsize_r'], ellipse['psfsize_min_r'], ellipse['psfsize_max_r'],
                    ellipse['psfsize_z'], ellipse['psfsize_min_z'], ellipse['psfsize_max_z']))
            html.write('</table>\n')
            #html.write('<br />\n')

        pngfile, thumbfile = '{}-custom-montage-grz.png'.format(galaxy1), 'thumb-{}-custom-montage-grz.png'.format(galaxy1)
        html.write('<p>Color mosaics showing the data (left panel), model (middle panel), and residuals (right panel).</p>\n')
        html.write('<table width="90%">\n')
        for bandsuffix in ('grz', 'FUVNUV', 'W1W2'):
            pngfile, thumbfile = '{}-custom-montage-{}.png'.format(galaxy1, bandsuffix), 'thumb-{}-custom-montage-{}.png'.format(galaxy1, bandsuffix)
            html.write('<tr><td><a href="{0}"><img src="{1}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile, thumbfile))
        html.write('</table>\n')

    def _html_ellipsefit_and_photometry(html, tractor, sample):
        html.write('<h2>Elliptical Isophote Analysis</h2>\n')
        if tractor is None:
            html.write('<p>Tractor catalog not available.</p>\n')
            html.write('<h3>Geometry</h3>\n')
            html.write('<h3>Photometry</h3>\n')
            return

        html.write('<h3>Geometry</h3>\n')
        html.write('<table>\n')
        html.write('<tr><th></th>\n')
        html.write('<th colspan="5">Tractor</th>\n')
        html.write('<th colspan="3">Ellipse Moments</th>\n')
        html.write('<th colspan="3">Surface Brightness<br /> Threshold Radii<br />(arcsec)</th>\n')
        html.write('<th colspan="3">Half-light Radii<br />(arcsec)</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>Galaxy</th>\n')
        html.write('<th>Type</th><th>n</th><th>r(50)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>Size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>R(24)</th><th>R(25)</th><th>R(26)</th>\n')
        html.write('<th>g(50)</th><th>r(50)</th><th>z(50)</th>\n')
        html.write('</tr>\n')

        for ss, tt in zip(sample, tractor):
            ee = np.hypot(tt['shape_e1'], tt['shape_e2'])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tt['shape_e2'], tt['shape_e1']) / 2))
            pa = pa % 180

            html.write('<tr><td>{}</td>\n'.format(ss[REFIDCOLUMN]))
            html.write('<td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                tt['type'], tt['sersic'], tt['shape_r'], pa, 1-ba))

            galaxyid = str(tt['ref_id'])
            ellipse = SGA.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                             galaxy_id=galaxyid, verbose=False)
            if bool(ellipse):
                html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    ellipse['sma_moment'], ellipse['pa_moment'], ellipse['eps_moment']))
                    #ellipse['majoraxis']*ellipse['refpixscale'], ellipse['pa_moment'], ellipse['eps_moment']))

                rr = []
                if 'sma_sb24' in ellipse.keys():
                    for rad in [ellipse['sma_sb24'], ellipse['sma_sb25'], ellipse['sma_sb26']]:
                        if rad < 0:
                            rr.append('...')
                        else:
                            rr.append('{:.3f}'.format(rad))
                    html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(rr[0], rr[1], rr[2]))
                else:
                    html.write('<td>...</td><td>...</td><td>...</td>\n')

                rr = []
                if 'cog_sma50_g' in ellipse.keys():
                    for rad in [ellipse['cog_sma50_g'], ellipse['cog_sma50_r'], ellipse['cog_sma50_z']]:
                        if rad < 0:
                            rr.append('...')
                        else:
                            rr.append('{:.3f}'.format(rad))
                    html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(rr[0], rr[1], rr[2]))
                else:
                    html.write('<td>...</td><td>...</td><td>...</td>\n')                
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<h3>Photometry</h3>\n')
        html.write('<table>\n')
        #html.write('<tr><th></th><th></th>\n')
        #html.write('<th colspan="3"></th>\n')
        #html.write('<th colspan="12">Curve of Growth</th>\n')
        #html.write('</tr>\n')
        html.write('<tr><th></th>\n')
        html.write('<th colspan="9">Tractor</th>\n')
        html.write('<th colspan="9">Curve of Growth</th>\n')
        #html.write('<th colspan="3">&lt R(24)<br />arcsec</th>\n')
        #html.write('<th colspan="3">&lt R(25)<br />arcsec</th>\n')
        #html.write('<th colspan="3">&lt R(26)<br />arcsec</th>\n')
        #html.write('<th colspan="3">Integrated</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>Galaxy</th>\n')
        html.write('<th>FUV</th><th>NUV</th><th>g</th><th>r</th><th>z</th><th>W1</th><th>W2</th><th>W3</th><th>W4</th>\n')
        html.write('<th>FUV</th><th>NUV</th><th>g</th><th>r</th><th>z</th><th>W1</th><th>W2</th><th>W3</th><th>W4</th>\n')
        html.write('</tr>\n')

        for tt, ss in zip(tractor, sample):
            fuv, nuv, g, r, z, w1, w2, w3, w4 = _get_mags(tt, pipeline=True)
            html.write('<tr><td>{}</td>\n'.format(ss[REFIDCOLUMN]))
            html.write('<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\n'.format(
                fuv, nuv, g, r, z, w1, w2, w3, w4))

            galaxyid = str(tt['ref_id'])
            ellipse = SGA.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                             galaxy_id=galaxyid, verbose=False)
            if bool(ellipse):# and 'cog_mtot_fuv' in ellipse.keys():
                #g, r, z = _get_mags(ellipse, R24=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                #g, r, z = _get_mags(ellipse, R25=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                #g, r, z = _get_mags(ellipse, R26=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                fuv, nuv, g, r, z, w1, w2, w3, w4 = _get_mags(ellipse, cog=True)                
                html.write('<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\n'.format(
                    fuv, nuv, g, r, z, w1, w2, w3, w4))
                #g, r, z = _get_mags(ellipse, cog=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')

        # Galaxy-specific mosaics--
        for igal in np.arange(len(tractor['ref_id'])):
            galaxyid = str(tractor['ref_id'][igal])
            #html.write('<h4>{}</h4>\n'.format(galaxyid))
            html.write('<h4>{}</h4>\n'.format(sample[REFIDCOLUMN][igal]))

            ellipse = SGA.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                             galaxy_id=galaxyid, verbose=verbose)
            if not bool(ellipse):
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')
                continue

            html.write('<table width="90%">\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband-FUVNUV.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband-FUVNUV.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="60%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="80%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband-W1W2.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband-W1W2.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="100%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')

            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-sbprofile.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            pngfile = '{}-custom-ellipse-{}-cog.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-sed.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')
            
            html.write('</table>\n')
            #html.write('<br />\n')

    def _html_ccd_diagnostics(html):
        html.write('<h2>CCD Diagnostics</h2>\n')

        html.write('<table width="90%">\n')
        pngfile = '{}-ccdpos.png'.format(galaxy1)
        html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
            pngfile))
        html.write('</table>\n')
        #html.write('<br />\n')
        
    # Read the catalogs and then build the page--
    nccds, tractor, sample = _read_ccds_tractor_sample(prefix='custom')

    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
        html.write('</style>\n')

        # Top navigation menu--
        html.write('<h1>{}</h1>\n'.format(galaxy1))
        raslice = SGA.io.get_raslice(gal[racolumn])
        html.write('<h4>RA Slice {}</h4>\n'.format(raslice))

        html.write('<a href="../../{}">Home</a>\n'.format(htmlhome))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))

        _html_group_properties(html, gal)
        _html_image_mosaics(html)
        _html_ellipsefit_and_photometry(html, tractor, sample)
        #_html_maskbits(html)
        #_html_ccd_diagnostics(html)

        html.write('<br /><br />\n')
        html.write('<a href="../../{}">Home</a>\n'.format(htmlhome))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
        html.write('<br />\n')

        html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
        html.write('<br />\n')
        html.write('</html></body>\n')



def make_html(sample=None, datadir=None, htmldir=None, bands=['g', 'r', 'i', 'z'],
              refband='r', region='dr11-south', pixscale=0.262, zcolumn='Z', intflux=None,
              racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
              first=None, last=None, galaxylist=None,
              mp=1, survey=None, makeplots=False,
              htmlhome='index.html', html_raslices=False,
              clobber=False, verbose=True, maketrends=False, ccdqa=False,
              args=None):
    """Make the HTML pages.

    """
    import subprocess
    from astrometry.util.multiproc import multiproc

    import SGA.io
    from SGA.coadds import _mosaic_width

    #datadir = SGA.io.sga_data_dir()
    #htmldir = SGA.io.sga_html_dir()
    datadir = os.path.join(SGA.io.sga_data_dir(), region)
    htmldir = os.path.join(SGA.io.sga_html_dir(), region)
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)

    if sample is None:
        sample = SGA.read_sample(first=first, last=last, galaxylist=galaxylist)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)

    # Only create pages for the set of galaxies with a montage.
    keep = np.arange(len(sample))
    _, missing, done, _ = SGA.io.missing_files(sample=sample, region=region, htmldir=htmldir,
                                               htmlindex=True)

    if len(done[0]) == 0:
        print('No galaxies with complete montages!')
        return

    print('Keeping {}/{} galaxies with complete montages.'.format(len(done[0]), len(sample)))
    sample = sample[done[0]]
    #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    # Build the home (index.html) page (always, irrespective of clobber)--
    build_htmlhome(sample, htmldir, htmlhome=htmlhome, pixscale=pixscale,
                   racolumn=racolumn, deccolumn=deccolumn, diamcolumn=diamcolumn,
                   html_raslices=html_raslices)

    # Now build the individual pages in parallel.
    if html_raslices:
        raslices = np.array([SGA.io.get_raslice(ra) for ra in sample[racolumn]])
        rasorted = np.argsort(raslices)
        galaxy, galaxydir, htmlgalaxydir = SGA.io.get_galaxy_galaxydir(sample[rasorted], region=region, html=True)
    else:
        rasorted = np.arange(len(sample))
        galaxy, galaxydir, htmlgalaxydir = SGA.io.get_galaxy_galaxydir(sample, region=region, html=True)

    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    mp = multiproc(nthreads=mp)
    args = []
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate(zip(
        sample[rasorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir))):
        args.append([ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                     racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy,
                     prevgalaxy, nexthtmlgalaxydir, prevhtmlgalaxydir, verbose,
                     clobber])
    ok = mp.map(_build_htmlpage_one, args)

    return 1
