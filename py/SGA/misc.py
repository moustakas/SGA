"""
SGA.misc
========

Miscellaneous utility code used by various scripts.

"""
import os, sys
import numpy as np


def viewer_inspect(cat, galaxycolname='GALAXY'):
    """Write a little catalog that can be uploaded to the viewer.

    """
    out = cat[galaxycolname, 'RA', 'DEC']
    out.rename_column(galaxycolname, 'NAME')
    outfile = os.path.join(os.getenv('HOME'), 'tmp', 'viewer.fits')
    print('Writing {} objects to {}'.format(len(cat), outfile))
    out.write(outfile, overwrite=True)


def imagetool_inspect(cat, group=False):
    """Write a little catalog that can be uploaded to
    https://yymao.github.io/decals-image-list-tool/

    """
    if group:
        galcol, racol, deccol = 'GROUP_NAME', 'GROUP_RA', 'GROUP_DEC'
    else:
        racol, deccol = 'RA', 'DEC'
        galcol = 'GALAXY'
        if not galcol in cat.colnames:
            galcol = 'NAME'
        
    outfile = os.path.join(os.getenv('HOME'), 'tmp', 'inspect.txt')
    print('Writing {} objects to {}'.format(len(cat), outfile))
    with open(outfile, 'w') as ff:
        ff.write('name ra dec\n')
        for ii, (gal, ra, dec) in enumerate(zip(cat[galcol], cat[racol], cat[deccol])):
            if gal.strip() == '':
                if 'ALTNAME' in cat.colnames:
                    gal = cat['ALTNAME'][ii].strip().replace(' ', '')
                    if gal == '':
                        gal = 'galaxy'
                else:
                    gal = 'galaxy'
            ff.write('{} {:.6f} {:.6f}\n'.format(gal, ra, dec))


def srcs2image(cat, wcs, band='r', allbands='grz', pixelized_psf=None, psf_sigma=1.0):
    """Build a model image from a Tractor catalog or a list of sources.

    issrcs - if True, then cat is already a list of sources.

    """
    import tractor, legacypipe, astrometry
    from legacypipe.catalog import read_fits_catalog

    if type(wcs) is tractor.wcs.ConstantFitsWcs or type(wcs) is legacypipe.survey.LegacySurveyWcs:
        shape = wcs.wcs.shape
    else:
        shape = wcs.shape
    model = np.zeros(shape)
    invvar = np.ones(shape)

    if pixelized_psf is None:
        vv = psf_sigma**2
        psf = tractor.GaussianMixturePSF(1.0, 0., 0., vv, vv, 0.0)
    else:
        psf = pixelized_psf

    tim = tractor.Image(model, invvar=invvar, wcs=wcs, psf=psf,
                        photocal=tractor.basics.LinearPhotoCal(1.0, band=band.lower()),
                        sky=tractor.sky.ConstantSky(0.0),
                        name='model-{}'.format(band))

    # Do we have a tractor catalog or a list of sources?
    if type(cat) is astrometry.util.fits.tabledata:
        srcs = legacypipe.catalog.read_fits_catalog(cat, bands=[band.lower()])
    else:
        srcs = cat

    tr = tractor.Tractor([tim], srcs)
    mod = tr.getModelImage(0)

    return mod


def simple_wcs(onegal, radius=None, factor=1.0, pixscale=0.262, zcolumn='Z'):
    '''Build a simple WCS object for a single galaxy.

    radius in pixels
    '''
    from astrometry.util.util import Tan

    if radius is None:
        if zcolumn in onegal.colnames:
            radius = 2 * cutout_radius_kpc(redshift=onegal[zcolumn], pixscale=pixscale)
        else:
            radius = 100 # hack! [pixels]
    
    diam = np.ceil(factor * 2 * radius).astype('int') # [pixels]
    simplewcs = Tan(onegal['RA'], onegal['DEC'], diam/2+0.5, diam/2+0.5,
                    -pixscale/3600.0, 0.0, 0.0, pixscale/3600.0, 
                    float(diam), float(diam))
    return simplewcs


def ccdwcs(ccd):
    '''Build a simple WCS object for a single CCD table.'''
    from astrometry.util.util import Tan

    W, H = ccd.width, ccd.height
    ccdwcs = Tan(*[float(xx) for xx in [ccd.crval1, ccd.crval2, ccd.crpix1,
                                        ccd.crpix2, ccd.cd1_1, ccd.cd1_2,
                                        ccd.cd2_1, ccd.cd2_2, W, H]])
    return W, H, ccdwcs


def arcsec2kpc(redshift, cosmo=None):
    """Compute and return the scale factor to convert a physical axis in arcseconds
    to kpc.

    """
    #cosmo = cosmology()
    return 1 / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/arcsec]


def statsinbins(xx, yy, binsize=0.1, minpts=10, xmin=None, xmax=None):
    """Compute various statistics in running bins along the x-axis.

    """
    from scipy.stats import binned_statistic

    # Need an exception if there are fewer than three arguments.
    if xmin == None:
        xmin = xx.min()
    if xmax == None:
        xmax = xx.max()

    nbin = int( (np.nanmax(xx) - np.nanmin(xx) ) / binsize )
    stats = np.zeros(nbin, [('xmean', 'f4'), ('xmedian', 'f4'), ('xbin', 'f4'),
                            ('npts', 'i4'), ('ymedian', 'f4'), ('ymean', 'f4'),
                            ('ystd', 'f4'), ('y25', 'f4'), ('y75', 'f4')])

    if False:
        def median(x):
            return np.nanmedian(x)

        def mean(x):
            return np.nanmean(x)

        def std(x):
            return np.nanstd(x)

        def q25(x):
            return np.nanpercentile(x, 25)

        def q75(x):
            return np.nanpercentile(x, 75)

        ystat, bin_edges, _ = binned_statistic(xx, yy, bins=nbin, statistic='median')
        stats['median'] = ystat

        bin_width = (bin_edges[1] - bin_edges[0])
        xmean = bin_edges[1:] - bin_width / 2

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic='mean')
        stats['mean'] = ystat

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic=std)
        stats['std'] = ystat

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic=q25)
        stats['q25'] = ystat

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic=q75)
        stats['q75'] = ystat

        keep = (np.nonzero( stats['median'] ) * np.isfinite( stats['median'] ))[0]
        xmean = xmean[keep]
        stats = stats[keep]
    else:
        _xbin = np.linspace(xmin, xmax, nbin)
        idx  = np.digitize(xx, _xbin)

        for kk in range(nbin):
            these = idx == kk
            npts = np.count_nonzero( yy[these] )

            stats['xbin'][kk] = _xbin[kk]
            stats['npts'][kk] = npts

            if npts > 0:
                stats['xmedian'][kk] = np.nanmedian( xx[these] )
                stats['xmean'][kk] = np.nanmean( xx[these] )

                stats['ystd'][kk] = np.nanstd( yy[these] )
                stats['ymean'][kk] = np.nanmean( yy[these] )

                qq = np.nanpercentile( yy[these], [25, 50, 75] )
                stats['y25'][kk] = qq[0]
                stats['ymedian'][kk] = qq[1]
                stats['y75'][kk] = qq[2]

        keep = stats['npts'] > minpts
        if np.count_nonzero(keep) == 0:
            return None
        else:
            return stats[keep]


def convert_tractor_e1e2(e1, e2):
    """Convert Tractor epsilon1, epsilon2 values to ellipticity and position angle.

    Taken from tractor.ellipses.EllipseE

    """
    e = np.hypot(e1, e2)
    ba = (1 - e) / (1 + e)
    #e = (ba + 1) / (ba - 1)

    phi = -np.rad2deg(np.arctan2(e2, e1) / 2)
    #angle = np.deg2rad(-2 * phi)
    #e1 = e * np.cos(angle)
    #e2 = e * np.sin(angle)

    return ba, phi
