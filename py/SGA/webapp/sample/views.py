"""
Holds the functions that send http responses to the browser, including 
rendering the html pages index.html, list.html, and centrals.html, or sending a download file.

All logic that must be done before the browser renders the html occurs here, including 
sessions, serialization, querying database, applying filters, and pagination.
"""

import os
import pickle
import tempfile
import numpy as np
import astropy.io.fits
from astropy.table import Table, Column
from django.shortcuts import render
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.http import HttpResponse
from .filters import SampleFilter
from .models import Sample

def list(req):
    """
    Returns the list.html download file, or renders the list.html page after it 
    applies the filter, stores result to session, and sets up pagination.
    
    Args:
        req: the http request
        
    Returns: 
        File stream if user clicked download, otherwise render for list.html
    """
    #if download button was pressed return a send_file
    if req.method == 'POST':
        # Ideally, this step would grab the full set of data from the parent
        # FITS file (before it's loaded into Django in load.py).  So this is a
        # hack!
        data = Table()
        for col in ('objid', 'ra', 'dec', 'morphtype', 'z',
                    'la', 'sdss_objid'):
            data[col] = [getattr(cc, col) for cc in cen_filtered]
            
        # Write the FITS file contents...
        f, tmpfn = tempfile.mkstemp(suffix='.fits')
        os.close(f)
        os.unlink(tmpfn)
        data.write(tmpfn)
        return send_file(tmpfn, 'image/fits', unlink=True, filename='results.fits')

    #otherwise render the page based on new filter
    #automatically sort by sga_id if no other sort value given
    sort = 'sga_id'
    if "sort" in req.GET:
        sort = req.GET.get('sort')

    #apply filter to centrals model, then store in queryset
    sample_filter = SampleFilter(req.GET, queryset=Sample.objects.all().order_by(sort))
    sample_filtered = sample_filter.qs
    #use pickle to serialize queryset, and store in session
    req.session['results_list'] = pickle.dumps(sample_filtered)
    #use django pagination functionality
    paginator = Paginator(sample_filtered, 50)
    page_num = req.GET.get('page')
    page = paginator.get_page(page_num)
    #include pagination values we will use in html page in the return statement
    return render(req, 'list.html', {'page': page, 'paginator': paginator})

    

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
    #include values we will use in html page in the return statement
    return render(req, 'centrals.html', {'cen_list': cen_list, 'index': index, 'cen': cen, 'next_index': next_index, 'prev_index': prev_index})

def send_file(fn, content_type, unlink=False, modsince=None, expires=3600, filename=None):
    """
    Creates a streaminghttpresponse to send download file to browser
    Taken from unwise.me views.py
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
