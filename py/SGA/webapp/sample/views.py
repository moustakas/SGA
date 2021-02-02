#!/usr/bin/env python

"""Holds the functions that send http responses to the browser, including
rendering the html pages index.html, list.html, and sample.html, or sending a
download file.

All logic that must be done before the browser renders the html occurs here,
including sessions, serialization, querying database, applying filters, and
pagination.

"""
import os, pickle, tempfile
import numpy as np

import astropy.io.fits
from astropy.table import Table, Column

from django.shortcuts import render
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.http import HttpResponse

from .filters import SampleFilter
from .models import Sample

def list(req):
    """Returns the list.html download file, or renders the list.html page after it
    applies the filter, stores result to session, and sets up pagination.
    
    Args:
        req: the http request
        
    Returns: 
        File stream if user clicked download, otherwise render for list.html

    """
    ##if download button was pressed return a send_file
    #if req.method == 'POST':
    #    # Ideally, this step would grab the full set of data from the parent
    #    # FITS file (before it's loaded into Django in load.py).  So this is a
    #    # hack!
    #    data = Table()
    #    for col in ('objid', 'ra', 'dec', 'morphtype', 'z',
    #                'la', 'sdss_objid'):
    #        data[col] = [getattr(cc, col) for cc in cen_filtered]
    #        
    #    # Write the FITS file contents...
    #    f, tmpfn = tempfile.mkstemp(suffix='.fits')
    #    os.close(f)
    #    os.unlink(tmpfn)
    #    data.write(tmpfn)
    #    return send_file(tmpfn, 'image/fits', unlink=True, filename='sga.fits')

    # Render the page based on new filter. Automatically sort by sga_id if no
    # other sort value given.
    sort = None
    if "sort" in req.GET:
        sort = req.GET.get('sort')

    print('list(): req.GET', req.GET)

    #queryset = Sample.objects.all()
    queryset = None
    
    cone_ra  = req.GET.get('conera','')
    cone_dec = req.GET.get('conedec','')
    cone_rad = req.GET.get('coneradius','')

    # save for form default
    cone_rad_arcmin = cone_rad
    if len(cone_ra) and len(cone_dec) and len(cone_rad):
        try:
            from django.db.models import F

            cone_ra = float(cone_ra)
            cone_dec = float(cone_dec)
            cone_rad = float(cone_rad) / 60.

            dd = np.deg2rad(cone_dec)
            rr = np.deg2rad(cone_ra)
            cosd = np.cos(dd)
            x, y, z = cosd * np.cos(rr), cosd * np.sin(rr), np.sin(dd)
            r2 = np.deg2rad(cone_rad)**2

            queryset = Sample.objects.all().annotate(
                r2=((F('ux')-x)*(F('ux')-x) +
                    (F('uy')-y)*(F('uy')-y) +
                    (F('uz')-z)*(F('uz')-z)))
            queryset = queryset.filter(r2__lt=r2)
            if sort is None:
                sort='r2'
            #queryset = sample_near_radec(cone_ra, cone_dec, cone_rad).order_by(sort)
        except ValueError:
            pass

    if queryset is None:
        queryset = Sample.objects.all()
    if sort is None:
        sort = 'sga_id'
        
    queryset = queryset.order_by(sort)

    #apply filter to Sample model, then store in queryset
    sample_filter = SampleFilter(req.GET, queryset)
    sample_filtered = sample_filter.qs

    #use pickle to serialize queryset, and store in session
    req.session['results_list'] = pickle.dumps(sample_filtered)

    #use django pagination functionality
    paginator = Paginator(sample_filtered, 50)
    page_num = req.GET.get('page')
    page = paginator.get_page(page_num)

    # Include pagination values we will use in html page in the return
    # statement.
    return render(req, 'list.html', {'page': page, 'paginator': paginator,
                                     'cone_ra':cone_ra, 'cone_dec':cone_dec,
                                     'cone_rad':cone_rad_arcmin})
    

def index(req):
    """
    Renders the homepage from index.html
    
    Args:
        req: the http request
        
    Returns: 
        Render for index.html
    
    """
    return render(req, 'index.html')

def centrals(req):
    """
    Renders the centrals.html page for the current index after it 
    loads queryset from session and determines previous and next index to look at.
    
    Args:
        req: the http request
        
    Returns: 
        Render for centrals.html based on index value

    """
    index = int(req.GET.get('index'))
    #load from session and use slicing to access info on that Central object
    cen_list = pickle.loads(req.session['results_list'])
    cen = cen_list[index-1:index][0]
    #determine previous and next index
    prev_index = index - 1
    if (prev_index == 0):
        prev_index = len(cen_list)
    next_index = index + 1
    if (next_index > len(cen_list)):
       next_index = 1
       
    # Include values we will use in html page in the return statement.
    return render(req, 'centrals.html', {'cen_list': cen_list, 'index': index, 'cen': cen,
                                         'next_index': next_index, 'prev_index': prev_index})

def send_file(fn, content_type, unlink=False, modsince=None, expires=3600, filename=None):
    """Creates a streaminghttpresponse to send download file to browser

    Taken from unwise.views.py.

    """
    import datetime
    from django.http import HttpResponseNotModified, StreamingHttpResponse

    '''
    modsince: If-Modified-Since header string from the client.
    '''
    st = os.stat(fn)
    f = open(fn, 'rb')
    if unlink:
        os.unlink(fn)
    # file was last modified.
    lastmod = datetime.datetime.fromtimestamp(st.st_mtime)

    if modsince:
        #print('If-modified-since:', modsince #Sat, 22 Nov 2014 01:12:39 GMT)
        ifmod = datetime.datetime.strptime(modsince, '%a, %d %b %Y %H:%M:%S %Z')
        #print('Parsed:', ifmod)
        #print('Last mod:', lastmod)
        dt = (lastmod - ifmod).total_seconds()
        if dt < 1:
            return HttpResponseNotModified()

    res = StreamingHttpResponse(f, content_type=content_type)
    # res['Cache-Control'] = 'public, max-age=31536000'
    res['Content-Length'] = st.st_size
    if filename is not None:
        res['Content-Disposition'] = 'attachment; filename="%s"' % filename
    # expires in an hour?
    now = datetime.datetime.utcnow()
    then = now + datetime.timedelta(0, expires, 0)
    timefmt = '%a, %d %b %Y %H:%M:%S GMT'
    res['Expires'] = then.strftime(timefmt)
    res['Last-Modified'] = lastmod.strftime(timefmt)
    return res

def sample_near_radec(ra, dec, rad, tablename='sample',
                      extra_where='', clazz=Sample):
    #from astrometry.util.starutil import deg2distsq
    dec = np.deg2rad(dec)
    ra = np.deg2rad(ra)
    cosd = np.cos(dec)
    x,y,z = cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)
    radius = rad + np.sqrt(2.)/2. * 2048 * 2.75 / 3600. * 1.01

    ## FIXME
    r2 = np.deg2rad(radius)**2
    #r2 = deg2distsq(radius)
    sample = clazz.objects.raw(
        ('SELECT *, ((ux-(%g))*(ux-(%g))+(uy-(%g))*(uy-(%g))+(uz-(%g))*(uz-(%g))) as r2'
         + ' FROM %s where r2 <= %g %s ORDER BY r2') %
        (x,x,y,y,z,z, tablename, r2, extra_where))
    
    return sample

