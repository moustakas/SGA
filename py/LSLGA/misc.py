"""
LSLGA.misc
==========

Miscellaneous code.

"""
import numpy as np

def plot_style(paper=False, talk=False):

    import seaborn as sns
    rc = {'font.family': 'serif'}#, 'text.usetex': True}
    #rc = {'font.family': 'serif', 'text.usetex': True,
    #       'text.latex.preamble': r'\boldmath'})
    palette = 'Set2'
    
    if paper:
        palette = 'deep'
        rc.update({'text.usetex': False})
    
    if talk:
        pass

    sns.set(style='ticks', font_scale=1.6, rc=rc)
    sns.set_palette(palette, 12)

    colors = sns.color_palette()
    #sns.reset_orig()

    return sns, colors

def custom_brickname(ra, dec):
    brickname = '{:06d}{}{:05d}'.format(
        int(1000*ra), 'm' if dec < 0 else 'p',
        int(1000*np.abs(dec)))
    return brickname

def is_point_in_desi(tiles, ra, dec, radius=None, return_tile_index=False):
    """If a point (`ra`, `dec`) is within `radius` distance from center of any
    tile, it is in DESI.

    Args:
        tiles (Table-like): The output of :func:`desimodel.io.load_tiles`, or
            a similar Table.
        ra (scalar or array-like): Right Ascension in degrees.
        dec (scalar or array-like): Declination in degrees.  The size of `dec`
            must match the size of `ra`.
        radius (float, optional): Tile radius in degrees;
            if `None` use :func:`desimodel.focalplane.get_tile_radius_deg`.
        return_tile_index (bool, optional): If ``True``, return the index of
            the nearest tile in tiles array.

    Returns:
        Return ``True`` if points given by `ra`, `dec` lie in the set of `tiles`.

    Notes:
        This function is optimized to query a lot of points.
    """
    from scipy.spatial import cKDTree as KDTree

    if radius is None:
        #radius = get_tile_radius_deg()
        radius = 1.606 # [deg]

    def _embed_sphere(ra, dec):
        """Embed `ra`, `dec` to a uniform sphere in three dimensions.
        """
        phi = np.radians(np.asarray(ra))
        theta = np.radians(90.0 - np.asarray(dec))
        r = np.sin(theta)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.cos(theta)
        return np.array((x, y, z)).T

    tilecenters = _embed_sphere(tiles['RA'], tiles['DEC'])
    tree = KDTree(tilecenters)
    # radius to 3d distance
    threshold = 2.0 * np.sin(np.radians(radius) * 0.5)
    xyz = _embed_sphere(ra, dec)
    d, i = tree.query(xyz, k=1)

    indesi = d < threshold
    if return_tile_index:
        return indesi, i
    else:
        return indesi


def radec2pix(nside, ra, dec):
    '''Convert `ra`, `dec` to nested pixel number.

    Args:
        nside (int): HEALPix `nside`, ``2**k`` where 0 < k < 30.
        ra (float or array): Right Accention in degrees.
        dec (float or array): Declination in degrees.

    Returns:
        Array of integer pixel numbers using nested numbering scheme.

    Notes:
        This is syntactic sugar around::

            hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)

        but also works with older versions of healpy that didn't have
        `lonlat` yet.
    '''
    import healpy as hp
    theta, phi = np.radians(90-dec), np.radians(ra)
    if np.isnan(np.sum(theta)) :
        raise ValueError("some NaN theta values")

    if np.sum((theta < 0)|(theta > np.pi))>0 :
        raise ValueError("some theta values are outside [0,pi]: {}".format(theta[(theta < 0)|(theta > np.pi)]))

    return hp.ang2pix(nside, theta, phi, nest=True)

def pix2radec(nside, pix):
    '''Convert nested pixel number to `ra`, `dec`.

    Args:
        nside (int): HEALPix `nside`, ``2**k`` where 0 < k < 30.
        ra (float or array): Right Accention in degrees.
        dec (float or array): Declination in degrees.

    Returns:
        Array of RA, Dec coorindates using nested numbering scheme. 

    Notes:
        This is syntactic sugar around::
            hp.pixelfunc.pix2ang(nside, pix, nest=True)
    
    '''
    import healpy as hp

    theta, phi = hp.pixelfunc.pix2ang(nside, pix, nest=True)
    ra, dec = np.degrees(phi), 90-np.degrees(theta)
    
    return ra, dec

def cosmology(WMAP=False, Planck=False):
    """Establish the default cosmology for the project."""

    if WMAP:
        from astropy.cosmology import WMAP9 as cosmo
    elif Planck:
        from astropy.cosmology import Planck15 as cosmo
    else:
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)        

    return cosmo

def arcsec2kpc(redshift):
    """Compute and return the scale factor to convert a physical axis in arcseconds
    to kpc.

    """
    cosmo = cosmology()
    return 1 / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/arcsec]

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

def ellipse_mask(xcen, ycen, semia, semib, phi, x, y):
    """Simple elliptical mask."""
    xp = (x-xcen) * np.cos(phi) + (y-ycen) * np.sin(phi)
    yp = -(x-xcen) * np.sin(phi) + (y-ycen) * np.cos(phi)
    return (xp / semia)**2 + (yp/semib)**2 <= 1


def srcs2image(srcs, wcs, psf_sigma=1.0):
    """Build a model image from a Tractor catalog.

    """
    from tractor import Tractor
    from tractor.image import Image
    from tractor.sky import ConstantSky
    from tractor.basics import LinearPhotoCal
    from tractor import GaussianMixturePSF

    try:
        shape = wcs.wcs.shape
    except:
        shape = wcs.shape
    model = np.zeros(shape)
    invvar = np.ones(shape)
    
    vv = psf_sigma**2
    psf = GaussianMixturePSF(1.0, 0., 0., vv, vv, 0.0)

    tim = Image(model, invvar=invvar, wcs=wcs, psf=psf,
                photocal=LinearPhotoCal(1.0, band='r'),
                sky=ConstantSky(0.0))

    tractor = Tractor([tim], srcs)
    mod = tractor.getModelImage(0)

    return mod

def ccdwcs(ccd):
    '''Build a simple WCS object for a single CCD table.'''
    W, H = ccd.width, ccd.height
    ccdwcs = Tan(*[float(xx) for xx in [ccd.crval1, ccd.crval2, ccd.crpix1,
                                        ccd.crpix2, ccd.cd1_1, ccd.cd1_2,
                                        ccd.cd2_1, ccd.cd2_2, W, H]])
    return W, H, ccdwcs

def simple_wcs(onegal, radius=100, factor=1.0, pixscale=0.262):
    '''Build a simple WCS object for a single galaxy.'''
    diam = np.ceil(factor * radius).astype('int') # [pixels]
    simplewcs = Tan(onegal['RA'], onegal['DEC'], diam/2+0.5, diam/2+0.5,
                    -pixscale/3600.0, 0.0, 0.0, pixscale/3600.0, 
                    float(diam), float(diam))
    return simplewcs
