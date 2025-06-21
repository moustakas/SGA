"""
SGA.html
========

Code to generate HTML output for the various stages of the SGA analysis.

"""
import os
import numpy as np


def get_layer(onegal):
    if onegal['DR'] == 'dr6':
        layer = 'mzls+bass-dr6'
    elif onegal['DR'] == 'dr7':
        layer = 'decals-dr5'
    else:
        print('Unrecognized data release {}!'.format(onegal['DR']))
        raise ValueError
    return layer


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

def _add_labels_one(args):
    """Wrapper function for the multiprocessing."""
    return add_labels_one(*args)

def add_labels_one(group, sample, overwrite=False, nothumb=False):

    jpgdir = os.path.join(SGAdir, 'cutouts', 'jpg')
    pngdir = os.path.join(SGAdir, 'cutouts', 'png')
    if not os.path.isdir(pngdir):
        os.mkdir(pngdir)

    groupname = get_groupname(group)
    galaxy = get_galaxy(group, sample, html=True)

    jpgfile = os.path.join(jpgdir, '{}.jpg'.format(groupname))
    pngfile = os.path.join(pngdir, '{}.png'.format(groupname))
    thumbfile = os.path.join(pngdir, 'thumb-{}.png'.format(groupname))
    
    if os.path.isfile(jpgfile):
        if os.path.isfile(pngfile) and not overwrite:
            print('File {} exists...skipping.'.format(pngfile))
        else:
            im = Image.open(jpgfile)
            sz = im.size
            fntsize = np.round(sz[0]/28).astype('int')
            width = np.round(sz[0]/175).astype('int')
            font = ImageFont.truetype(fonttype, size=fntsize)
            draw = ImageDraw.Draw(im)

            # Label the group--
            draw.text((0+fntsize*2, 0+fntsize*2), galaxy, font=font)
    
            # Add a scale bar--
            x0, x1, yy = sz[1]-fntsize*2-barlen, sz[1]-fntsize*2, sz[0]-fntsize*2
            draw.line((x0, yy, x1, yy), fill='white', width=width)
            im.save(pngfile)    
        
            # Generate a thumbnail
            if not nothumb:
                cmd = 'convert -thumbnail 300x300 {} {}'.format(pngfile, thumbfile)
                os.system(cmd)

def add_labels(groupsample, sample, overwrite=False):
    labelargs = list()
    for group in groupsample:
        labelargs.append((group, sample, overwrite))
    if mp > 1:
        p = multiprocessing.Pool(mp)
        res = p.map(_add_labels_one, labelargs)
        p.close()
    else:
        for args in labelargs:
            res = _add_labels_one(args)

def html_rows(_groupkeep, sample, nperrow=4):
    
    # Not all objects may have been analyzed.
    these = [os.path.isfile(os.path.join(SGAdir, 'cutouts', 'png', '{}.png'.format(
        get_groupname(gg)))) for gg in _groupkeep]
    groupkeep = _groupkeep[these]
    
    nrow = np.ceil(len(groupkeep) / nperrow).astype('int')
    groupsplit = list()
    for ii in range(nrow):
        i1 = nperrow*ii
        i2 = nperrow*(ii+1)
        if i2 > len(groupkeep):
            i2 = len(groupkeep)
        groupsplit.append(groupkeep[i1:i2])
    print('Splitting the sample into {} rows with {} mosaics per row.'.format(nrow, nperrow))

    html.write('<table class="ls-gallery">\n')
    html.write('<tbody>\n')
    for grouprow in groupsplit:
        html.write('<tr>\n')
        for group in grouprow:
            groupname = get_groupname(group)
            galaxy = get_galaxy(group, sample, html=True)

            pngfile = os.path.join('cutouts', 'png', '{}.png'.format(groupname))
            thumbfile = os.path.join('cutouts', 'png', 'thumb-{}.png'.format(groupname))
            img = 'src="{}" alt="{}"'.format(thumbfile, galaxy)
            #img = 'class="ls-gallery" src="{}" alt="{}"'.format(thumbfile, nicename)
            html.write('<td><a href="{}"><img {}></a></td>\n'.format(pngfile, img))
        html.write('</tr>\n')
        html.write('<tr>\n')
        for group in grouprow:
            groupname = get_groupname(group)
            galaxy = '{}: {}'.format(groupname.upper(), get_galaxy(group, sample, html=True))
            layer = get_layer(group)
            href = '{}/?layer={}&ra={:.8f}&dec={:.8f}&zoom=12'.format(viewerurl, layer, group['ra'], group['dec'])
            html.write('<td><a href="{}" target="_blank">{}</a></td>\n'.format(href, galaxy))
        html.write('</tr>\n')
    html.write('</tbody>\n')            
    html.write('</table>\n')


def make_plots(sample, analysisdir=None, htmldir='.', refband='r',
               band=('g', 'r', 'z'), overwrite=False, verbose=True):
    """Make QA plots.

    """
    sample_trends(sample, htmldir, analysisdir=analysisdir, verbose=verbose)

    for gal in sample:
        objid, objdir = get_objid(gal, analysisdir=analysisdir)

        htmlobjdir = os.path.join(htmldir, '{}'.format(objid))
        if not os.path.isdir(htmlobjdir):
            os.makedirs(htmlobjdir, exist_ok=True)

        # Build the ellipse plots.
        qa_ellipse_results(objid, objdir, htmlobjdir, band=band,
                           overwrite=overwrite, verbose=verbose)

        qa_sersic_results(objid, objdir, htmlobjdir, band=band,
                          overwrite=overwrite, verbose=verbose)

        # Build the montage coadds.
        qa_montage_coadds(objid, objdir, htmlobjdir, overwrite=overwrite, verbose=verbose)

        # Build the MGE plots.
        #qa_mge_results(objid, objdir, htmlobjdir, refband='r', band=band,
        #               overwrite=overwrite, verbose=verbose)

def _javastring():
    """Return a string that embeds a date in a webpage."""
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
        
def make_html(sample=None, htmldir=None, dr='dr6-dr7', makeplots=True, overwrite=False,
              verbose=True):
    """Make the HTML pages.

    """
    import SGA.io

    if htmldir is None:
        htmldir = SGA.io.html_dir()

    sample = SGA.io.read_parent(dr=dr)
    objid, objdir = legacyhalos.io.get_objid(sample)

    reject = []
    toss = np.zeros(len(groupsample), dtype=bool)
    for ii, gg in enumerate(groupsample['groupid']):
        for rej in np.atleast_1d(reject):
            toss[ii] = rej in gg.lower()
            if toss[ii]:
                break
    print('Rejecting {} groups.'.format(np.sum(toss)))
    groupkeep = groupsample[~toss]
    if np.sum(toss) > 0:
        grouprej = groupsample[toss]
    else:
        grouprej = []

    # Write the last-updated date to a webpage.
    js = _javastring()       

    # Get the viewer link
    def _viewer_link(gal, dr):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = 2 * cutout_radius_150kpc(redshift=gal['z'], pixscale=0.262) # [pixels]
        if width > 400:
            zoom = 14
        else:
            zoom = 15
        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=decals-{}'.format(
            baseurl, gal['ra'], gal['dec'], zoom, dr)
        return viewer

    homehtml = 'index.html'

    # Build the home (index.html) page--
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)
    htmlfile = os.path.join(htmldir, homehtml)

    with open(htmlfile, 'w') as html:
        html.write('<html><head>\n')
        html.write('<style type="text/css">\n')
        html.write('table.ls-gallery {width: 90%;}\n')
        html.write('p.ls-gallery {width: 80%;}\n')
        html.write('</style>\n')
        html.write('</head><body>\n')
        html.write('<h1>Siena Galaxy Atlas 2020 (SGA-2020)</h1>\n')
        html.write("""<p class="ls-gallery">Each thumbnail links to a larger image while the galaxy 
        name below each thumbnail links to the <a href="http://legacysurvey.org/viewer">Sky Viewer</a>.  
        For reference, the horizontal white bar in the lower-right corner of each image represents 
        one arcminute.</p>\n""")
        html_rows(groupkeep, sample)
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</body></html>\n')

    if makeplots:
        make_plots(sample, analysisdir=analysisdir, htmldir=htmldir, refband=refband,
                   band=band, overwrite=overwrite, verbose=verbose)
