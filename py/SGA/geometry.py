"""
SGA.geometry
============

Geometry (mostly ellipse-based) function.

"""
import pdb
import numpy as np
from astropy.table import Table

from SGA.logger import log


class EllipseProperties:
    """
    Fit an ellipse to the flux distribution of the largest labelled blob in a 2D image.

    After calling `fit(image, method, percentile, smooth_sigma)`, attributes are:
      x0, y0         # flux-weighted centroid (pixels)
      ba             # minor-to-major axis ratio (b/a)
      pa             # astronomical PA (deg CCW from +y, range 0–180)
      a              # chosen major-axis length (pixels)
      a_rms          # RMS-based semi-major axis (pixels)
      a_percentile   # percentile-based radius (pixels)
      labels         # label array for all blobs
      blob_mask      # mask of the selected largest blob

    """
    def __init__(self):
        self.x0 = None
        self.y0 = None
        self.ba = None
        self.pa = None
        self.a = None
        self.a_rms = None
        self.a_percentile = None
        self.labels = None
        self.blob_mask = None

    def fit(self, image, mask=None, method='percentile', percentile=0.95,
            smooth_sigma=1.0):
        """
        Label and smooth the image, then select the largest contiguous blob
        and compute ellipse properties using second moments.

        Parameters
        ----------
        image : 2D ndarray
            Pixel flux values (background should be zero or masked).
        method : {'percentile', 'rms'}
            Major-axis estimator:
            - 'percentile': brightness-weighted percentile radius
            - 'rms':        root of largest eigenvalue (second moment)
        percentile : float
            Fraction (0 < percentile <= 1) for percentile radius when method='percentile'.
        smooth_sigma : float, default=1.0
            Gaussian sigma (pixels) for smoothing before blob detection.

        Returns
        -------
        self : EllipseProperties

        """
        from scipy import ndimage

        # 1) optionally smooth for blob detection
        if smooth_sigma and smooth_sigma > 0:
            smoothed = ndimage.gaussian_filter(image, sigma=smooth_sigma)
        else:
            smoothed = image

        # 2) label positive pixels in smoothed image, pick largest blob
        if mask is None:
            mask = (smoothed > 0)

        labels, nblobs = ndimage.label(mask)
        self.labels = labels
        if nblobs < 1:
            raise ValueError("No positive blobs found in image.")
        sizes = ndimage.sum(mask, labels, index=np.arange(1, nblobs+1))
        largest = np.argmax(sizes) + 1
        self.blob_mask = (labels == largest)
        blob_idx = np.flatnonzero(self.blob_mask)

        # 3) pixel coordinates and smoothed fluxes for blob pixels
        yy, xx = np.indices(image.shape)
        x_sel = xx.flat[blob_idx]
        y_sel = yy.flat[blob_idx]
        flux = smoothed.flat[blob_idx]
        if np.any(flux < 0):
            log.warning('Negative flux in image!')
            import pdb ; pdb.set_trace()
        F = flux.sum()

        # 4) flux-weighted centroid
        self.x0 = np.dot(flux, x_sel) / F
        self.y0 = np.dot(flux, y_sel) / F

        # 5) central second moments
        dx = x_sel - self.x0
        dy = y_sel - self.y0
        Mxx = np.dot(flux, dx*dx) / F
        Myy = np.dot(flux, dy*dy) / F
        Mxy = np.dot(flux, dx*dy) / F

        # 6) diagonalize inertia tensor
        eigvals, eigvecs = np.linalg.eigh([[Mxx, Mxy], [Mxy, Myy]])
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # 7) RMS-based semi-major axis
        self.a_rms = np.sqrt(eigvals[0])

        # 8) percentile-based radius
        r = np.sqrt(dx*dx + dy*dy)
        order_r = np.argsort(r)
        cumflux = np.cumsum(flux[order_r])
        cumfrac = cumflux / cumflux[-1]
        self.a_percentile = float(np.interp(percentile, cumfrac, r[order_r]))

        # 9) select final major axis
        if method == 'rms':
            self.a = self.a_rms
        elif method == 'percentile':
            self.a = self.a_percentile
        else:
            raise ValueError(f"Unknown method: {method}")

        # 10) compute axis ratio and astronomical PA within 0-180°
        b = np.sqrt(eigvals[1])
        self.ba = b / self.a_rms if self.a_rms > 0 else 0
        vx, vy = eigvecs[:, 0]
        pa_cart = np.degrees(np.arctan2(vy, vx)) % 180.0
        self.pa = (pa_cart - 90.) % 180.0

        return self

    def plot(self, image=None, ax=None, imshow_kwargs=None,
             ellipse_kwargs=None, blob_outline_kwargs=None):
        """
        Display the provided image (or blob mask) with the outline of the selected largest blob
        and overlay the fitted ellipse.

        Parameters
        ----------
        image : 2D ndarray, optional
            Image to display. If None, uses the blob mask.
        ax : matplotlib.axes.Axes, optional
            Ax for plotting. If None, a new figure/axes is created.
        imshow_kwargs : dict, optional
            Passed to ax.imshow (e.g., cmap, origin).
        ellipse_kwargs : dict, optional
            Passed to the Ellipse patch (e.g., edgecolor).
        blob_outline_kwargs : dict, optional
            Passed to ax.contour for the largest blob outline (e.g., colors, linewidths).

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        if ax is None:
            fig, ax = plt.subplots()
        if imshow_kwargs is None:
            imshow_kwargs = {'origin': 'lower', 'cmap': 'cividis'}
        disp = image if image is not None else self.blob_mask
        ax.imshow(disp, **imshow_kwargs)

        # outline only the largest blob
        if blob_outline_kwargs is None:
            blob_outline_kwargs = {'colors': 'k', 'linewidths': 1, 'alpha': 0.7}
        ax.contour(self.blob_mask, levels=[0.5], **blob_outline_kwargs)

        # overlay ellipse
        if ellipse_kwargs is None:
            ellipse_kwargs = {'edgecolor': 'red', 'facecolor': 'none'}
        angle = self.pa - 90.
        ell = Ellipse((self.x0, self.y0), 2*self.a, 2*self.a*self.ba,
                      angle=angle, **ellipse_kwargs)
        ax.add_patch(ell)

        # centroid marker
        ax.plot(self.x0, self.y0, marker='+', color=ellipse_kwargs.get('edgecolor','red'))
        return ax


def in_ellipse_mask(xcen, ycen, semia, semib, pa, x, y):
    """Simple elliptical mask using astronomical PA convention.

    Parameters
    ----------
    xcen, ycen : float
        Center of the ellipse.
    semia, semib : float
        Semi-major and semi-minor axes lengths.
    pa : float
        Position angle in degrees, measured CCW from the +y (north) axis.
    x, y : array-like
        Coordinates to test.

    Returns
    -------
    mask : ndarray of bool
        True for points inside or on the ellipse.

    Note
    ----

    If generating an image of size `size, `x` and `y` should be
    defined as:
        ```
        xgrid, ygrid = np.meshgrid(np.arange(size), np.arange(size), indexing='xy')
        ```

    """
    # convert PA to radians
    theta = np.deg2rad(pa)
    dx = x - xcen
    dy = y - ycen

    # Major-axis direction vector = (sin θ, cos θ)
    xp =  dx * np.sin(theta) + dy * np.cos(theta)
    # Minor-axis (90° CCW from major) = (–cos θ, sin θ)
    yp = -dx * np.cos(theta) + dy * np.sin(theta)

    return (xp/semia)**2 + (yp/semib)**2 <= 1


def get_tractor_ellipse(r50, e1, e2):
    """Convert Tractor epsilon1, epsilon2 values to ellipticity and position angle.

    Taken in part from tractor.ellipses.EllipseE.

    r50 in arcsec

    """
    e = np.hypot(e1, e2)
    ba = (1. - e) / (1. + e)
    #e = (ba + 1.) / (ba - 1.)

    phi = -np.rad2deg(np.arctan2(e2, e1) / 2.)
    #angle = np.deg2rad(-2 * phi)
    #e1 = e * np.cos(angle)
    #e2 = e * np.sin(angle)

    pa = (180. - phi) % 180
    diam = r50 * 2. * 1.2 # [radius-->diameter then 20% higher]

    return diam, ba, pa


def get_basic_geometry(cat, galaxy_column='OBJNAME', verbose=False):
    """From a catalog containing magnitudes, diameters, position angles, and
    ellipticities, return a "basic" value for each property.

    Priority order: RC3, TWOMASS, SDSS, ESO, NED/BASIC

    """
    nobj = len(cat)

    basic = Table()
    basic['GALAXY'] = cat[galaxy_column].value

    # default
    magcol = 'MAG_LIT'
    diamcol = 'DIAM_LIT'

    # HyperLeda
    if 'LOGD25' in cat.columns:
        ref = 'HYPERLEDA'
        magcol = f'MAG_{ref}'
        diamcol = f'DIAM_{ref}'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'BT'
                band = 'B'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'LOGD25'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = 0.1 * 10.**cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'LOGR25'
                I = ~np.isnan(cat[col]) * (cat[col] != 0.)
                if np.sum(I) > 0:
                    val[I] = 10.**(-cat[col][I])
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_{ref}'] = val
            basic[f'{prop.upper()}_{ref}_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_{ref}'] = val_band

    # SGA2020
    elif 'D26' in cat.columns:
        ref = 'SGA2020'
        magcol = f'MAG_{ref}'
        diamcol = f'DIAM_{ref}'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'R_MAG_SB26'
                band = 'R'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'D26'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'BA'
                I = ~np.isnan(cat[col]) * (cat[col] != 0.)
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_{ref}'] = val
            basic[f'{prop.upper()}_{ref}_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_{ref}'] = val_band

    # LVD
    elif 'RHALF' in cat.columns:
        ref = 'LVD'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'APPARENT_MAGNITUDE_V'
                band = 'V'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'RHALF' # [arcmin]
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    # see analyze-lvd
                    val[I] = cat[col][I] * 1.2 * 2. # half-light-->full-light; radius-->diameter
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'ELLIPTICITY' # =1-b/a
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = 1. - cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'POSITION_ANGLE'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I] % 180 # put in the range [0, 180]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

    # DR9/DR10 supplement
    elif 'SHAPE_R' in cat.columns:
        ref = 'DR910'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'FLUX_R'
                band = 'r'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = 22.5 - 2.5 * np.log10(cat[col][I])
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'SHAPE_R'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = 2. * 1.2 * cat[col][I] / 60. # [arcmin]
                    val_ref[I] = ref
            elif prop == 'ba' or prop == 'pa':
                I = ~np.isnan(cat['SHAPE_E1']) * ~np.isnan(cat['SHAPE_E2'])
                if np.sum(I) > 0:
                    from SGA.geometry import get_tractor_ellipse
                    _, ba, pa = get_tractor_ellipse(cat['SHAPE_R'][I], cat['SHAPE_E1'][I], cat['SHAPE_E2'][I])
                    if prop == 'ba':
                        val[I] = ba
                    elif prop == 'pa':
                        val[I] = pa
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

    # custom
    elif 'DIAM' in cat.columns:
        ref = 'CUSTOM'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'MAG'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = cat[f'{col}_BAND'][I]
            elif prop == 'diam':
                col = 'DIAM' # [arcmin]
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'BA'
                I = cat[col] != -99.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = cat[col] != -99.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band
    # NED
    else:
        for prop in ('mag', 'diam', 'ba', 'pa'):
            if prop == 'mag':
                refs = ('SDSS', 'TWOMASS', 'RC3')
                bands = ('R', 'K', 'B')
            else:
                refs = ('ESO', 'SDSS', 'TWOMASS', 'RC3')
                bands = ('B', 'R', 'K', 'B')
            nref = len(refs)

            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            #allI = np.zeros((nobj, nref), bool)
            for iref, (ref, band) in enumerate(zip(refs, bands)):
                if prop == 'mag':
                    col = f'{ref}_{band}'
                else:
                    col = f'{ref}_{prop.upper()}_{band}'
                I = cat[col] > 0.
                #allI[:, iref] = I

                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

        # supplement any missing values with the "BASIC" data
        I = (basic['MAG_LIT'] <= 0.) * (cat['BASIC_MAG'] > 0.)
        if np.any(I):
            basic['MAG_LIT'][I] = cat['BASIC_MAG'][I]
            basic['BAND_LIT'][I] = 'V'

        I = (basic['DIAM_LIT'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.)
        if np.any(I):
            basic['DIAM_LIT'][I] = cat['BASIC_DMAJOR'][I]
            basic['DIAM_LIT_REF'][I] = 'BASIC'

        I = (basic['BA_LIT'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.) * (cat['BASIC_DMINOR'] > 0.)
        if np.any(I):
            basic['BA_LIT'][I] = cat['BASIC_DMINOR'][I] / cat['BASIC_DMAJOR'][I]
            basic['BA_LIT_REF'][I] = 'BASIC'

    # summarize
    if verbose:
        M = basic[magcol] > 0.
        D = basic[diamcol] > 0.
        log.info(f'Derived photometry for {np.sum(M):,d}/{nobj:,d} objects and ' + \
                 f'diameters for {np.sum(D):,d}/{nobj:,d} objects.')

    return basic


def parse_geometry(cat, ref, mindiam=152*0.262):
    """Parse a specific set of elliptical geometry.

    ref - choose from among SGA2020, HYPERLEDA, RC3, LVD, SMUDGes, or LIT

    """
    nobj = len(cat)
    diam = np.zeros(nobj) - 99. # [arcsec]
    ba = np.ones(nobj)
    pa = np.zeros(nobj)
    outref = np.zeros(nobj, '<U9')

    if ref == 'SGA2020':
        I = cat['DIAM_SGA2020'] > 0.
        if np.any(I):
            diam[I] = cat['DIAM_SGA2020'][I] * 60. # [arcsec]
            ba[I] = cat['BA_SGA2020'][I]
            pa[I] = cat['PA_SGA2020'][I]
            outref[I] = ref
    elif ref == 'HYPERLEDA':
        I = cat['DIAM_HYPERLEDA'] > 0.
        if np.any(I):
            diam[I] = cat['DIAM_HYPERLEDA'][I] * 60. # [arcsec]
            ba[I] = cat['BA_HYPERLEDA'][I]
            pa[I] = cat['PA_HYPERLEDA'][I]
            outref[I] = ref
    elif ref == 'LIT':
        I = cat['DIAM_LIT'] > 0.
        if np.any(I):
            diam[I] = cat['DIAM_LIT'][I] * 60. # [arcsec]
            ba[I] = cat['BA_LIT'][I]
            pa[I] = cat['PA_LIT'][I]
            outref[I] = cat['DIAM_LIT_REF']

    #I = diam <= 0.
    #if np.any(I):
    #    diam[I] = mindiam # [arcsec]
    #    outref[I] = 'NONE'

    # clean up missing values of BA and PA
    ba[ba < 0.] = 1.
    pa[pa < 0.] = 0.

    if nobj == 1:
        return diam[0], ba[0], pa[0], outref[0]
    else:
        return diam, ba, pa, outref


def choose_geometry(cat, mindiam=152*0.262, get_mag=False):
    """Choose an object's geometry, selecting between the
    NED-assembled (literature) values (DIAM, BA, PA), values from the
    SGA2020 (DIAM_SGA2020, BA_SGA2020, PA_SGA2020), and HyperLeda's
    values (DIAM_HYPERLEDA, BA_HYPERLEDA, PA_HYPERLEDA).

    mindiam is ~40 arcsec

    Default values of BA and PA are 1.0 and 0.0.
    Default value of mag is 18.

    """
    # Is this a parent catalog with only one set of geometry
    # measurements? If so, take the values and run!
    if 'DIAM' in cat.colnames and 'BA' in cat.colnames and 'PA' in cat.colnames:
        diam = cat['DIAM'].value * 60.
        ba = cat['BA'].value
        pa = cat['PA'].value
        ref = np.array(['parent'] * len(cat))
        if get_mag:
            if 'MAG' in cat.colnames and 'BAND':
                mag = cat['MAG'].value
                band = cat['BAND'].value
            return diam, ba, pa, ref, mag, band
        else:
            return diam, ba, pa, ref


    nobj = len(cat)
    diam = np.zeros(nobj) - 99.
    ba = np.zeros(nobj) - 99.
    pa = np.zeros(nobj) - 99.
    ref = np.zeros(nobj, '<U9')

    # always prefer LVD because they were all visually determined and
    # inspected
    I = (cat['DIAM_LIT_REF'].value == 'LVD') * (diam == -99.)
    if np.any(I):
        diam[I] = cat['DIAM_LIT'][I] * 60.
        ba[I] = cat['BA_LIT'][I]
        pa[I] = cat['PA_LIT'][I]
        ref[I] = 'LVD'

    # take the largest diameter
    datarefs = np.array(['SGA2020', 'HYPERLEDA', 'LIT'])
    dataindx = np.argmax((cat['DIAM_SGA2020'].value, cat['DIAM_HYPERLEDA'].value, cat['DIAM_LIT'].value), axis=0)

    # first require all of diam, ba, pa...
    for iref, dataref in enumerate(datarefs):
        I = ((dataindx == iref) * (diam == -99.) * (ba == -99.) * (pa == -99.) *
             (cat[f'DIAM_{dataref}'] != -99.) * (cat[f'BA_{dataref}'] != -99.) *
             (cat[f'PA_{dataref}'] != -99.))
        if np.any(I):
            diam[I] = cat[f'DIAM_{dataref}'][I] * 60.
            ba[I] = cat[f'BA_{dataref}'][I]
            pa[I] = cat[f'PA_{dataref}'][I]
            ref[I] = datarefs[iref]
            # special-case LVD, RC3, and SMUDGes
            if dataref == 'LIT':
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'SMUDGes')[0]
                if len(J) > 0:
                    ref[I][J] = 'SMUDGes'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'LVD')[0]
                if len(J) > 0:
                    ref[I][J] = 'LVD'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'RC3')[0]
                if len(J) > 0:
                    ref[I][J] = 'RC3'

    # ...and then just diam.
    for iref, dataref in enumerate(datarefs):
        I = (dataindx == iref) * (diam == -99.) * (cat[f'DIAM_{dataref}'] != -99.)
        if np.any(I):
            diam[I] = cat[f'DIAM_{dataref}'][I] * 60.
            ba[I] = cat[f'BA_{dataref}'][I]
            pa[I] = cat[f'PA_{dataref}'][I]
            ref[I] = datarefs[iref]
            # special-case LVD, RC3, and SMUDGes
            if dataref == 'LIT':
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'SMUDGes')[0]
                if len(J) > 0:
                    ref[I][J] = 'SMUDGes'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'LVD')[0]
                if len(J) > 0:
                    ref[I][J] = 'LVD'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'RC3')[0]
                if len(J) > 0:
                    ref[I][J] = 'RC3'

    # missing diameters
    I = diam <= 0.
    if np.any(I):
        ref[I] = 'NONE'

    # set a minimum floor on the diameter
    I = diam <= mindiam
    if np.any(I):
        diam[I] = mindiam

    # clean up missing values of BA and PA
    ba[ba < 0.] = 1.
    pa[pa < 0.] = 0.

    if get_mag:
        mag = np.zeros(nobj) - 99.
        band = np.zeros(nobj, '<U1')
        for magref in ['SGA2020', 'HYPERLEDA', 'LIT']:
            I = (mag == -99.) * (cat[f'MAG_{magref}'] != -99.)
            #print(magref, np.sum(I))
            if np.any(I):
                mag[I] = cat[f'MAG_{magref}'][I]
                band[I] = cat[f'BAND_{magref}'][I]

        I = (mag == -99.)
        if np.any(I):
            mag[I] = 18.
            #band[I] = ''

    ## return scalars
    #if nobj == 1:
    #    diam = diam[0]
    #    ba = ba[0]
    #    pa = pa[0]
    #    ref = ref[0]

    if get_mag:
        return diam, ba, pa, ref, mag, band
    else:
        return diam, ba, pa, ref
