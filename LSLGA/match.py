"""
LSLGA.match
===========

Code to match to various external catalogs, including the Legacy Surveys imaging.

"""

import numpy as np

def get_masked_pixels(ccds, survey, wcs, debug=False):
    """Given an (input) set of CCDS (restricted, one supposes, to a large-galaxy
    cutout), compute the fraction of interpolated and saturated pixels.

    """
    nccd = len(ccds)
    W, H = wcs.get_width(), wcs.get_height()
    radecpoly = np.array([wcs.pixelxy2radec(x,y) for 
                          x, y in [(1, 1), (W, 1), (W, H), (1, H), (1, 1)]])
    for ii, ccd in enumerate(ccds):
        im = survey.get_image_object(ccd)
        x0, x1, y0, y1, slc = im.get_image_extent(wcs=im.get_wcs(), radecpoly=radecpoly)
        dq = im.read_dq(slice=slc)
        ccds.galaxy_npix[ii] = dq.size
        ccds.galaxy_fracsatur[ii] = np.sum(dq == 4) / dq.size  # saturated
        ccds.galaxy_fracinterp[ii] = np.sum(dq == 8) / dq.size # interpolated
        if debug:
            print(ii, ccd.filter, 100 * ccds.galaxy_fracsatur[ii], 100 * ccds.galaxy_fracinterp[ii])
            plt.imshow( (dq & 8 != 0), origin='lower')
            plt.show()
            
    return ccds

def simple_wcs(onegal, diam, PIXSCALE=0.262):
    """Build a simple WCS object for a single galaxy.

    """
    from astrometry.util.util import Tan
    size = np.rint(diam * 60 / PIXSCALE).astype('int') # [pixels]
    wcs = Tan(onegal['ra'], onegal['dec'], size/2+0.5, size/2+0.5,
                 -PIXSCALE/3600.0, 0.0, 0.0, PIXSCALE/3600.0, 
                 float(size), float(size))
    
    return wcs

