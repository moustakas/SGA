# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=============
desiutil.dust
=============

Get :math:`E(B-V)` values from the `Schlegel, Finkbeiner & Davis (1998; SFD98)`_ dust map.

.. _`Schlegel, Finkbeiner & Davis (1998; SFD98)`: http://adsabs.harvard.edu/abs/1998ApJ...500..525S.
"""
import os
import numpy as np
import itertools
from astropy.io.fits import getdata
from astropy.coordinates import SkyCoord
from astropy import units as u
#from desiutil.log import get_logger
#log = get_logger()

def extinction_total_to_selective_ratio(band, photsys, match_legacy_surveys=False) :
    """Return the linear coefficient R_X = A(X)/E(B-V) where
    A(X) = -2.5*log10(transmission in X band),
    for band X in 'G','R' or 'Z' when
    photsys = 'N' or 'S' specifies the survey (BASS+MZLS or DECALS),
    or for band X in 'G', 'BP', 'RP' when photsys = 'G' (when gaia dr2)
    or for band X in 'W1', 'W2', 'W3', 'W4' when photsys is either 'N' or 'S'
    E(B-V) is interpreted as SFD.

    Args:
        band : 'G', 'R', 'Z', 'BP', 'RP', 'W1', 'W2', 'W3', or 'W4'
        photsys : 'N' or 'S'

    Returns:
        scalar, total extinction A(band) = -2.5*log10(transmission(band))
    """
    if match_legacy_surveys :
        # Based on the fit from the columns MW_TRANSMISSION_X and EBV
        # for the DR8 target catalogs and propagated in fibermaps
        # R_X = -2.5*log10(MW_TRANSMISSION_X) / EBV
        # It is the same value for the N and S surveys in DR8 and DR9 catalogs.
        # see also https://github.com/dstndstn/tractor/issues/99
        R={"G_N":3.2140,
           "R_N":2.1650,
           "Z_N":1.2110,
           "G_S":3.2140,
           "R_S":2.1650,
           "Z_S":1.2110,
           "G_G":2.512,
           "BP_G":3.143,
           "RP_G":1.663,
        }
    else :
        # From https://desi.lbl.gov/trac/wiki/ImagingStandardBandpass
        # DECam u  3881.6   3.994
        # DECam g  4830.8   3.212
        # DECam r  6409.0   2.164
        # DECam i  7787.5   1.591
        # DECam z  9142.7   1.211
        # DECam Y  9854.5   1.063
        # BASS g  4772.1   3.258
        # BASS r  6383.6   2.176
        # MzLS z  9185.1   1.199
        # Consistent with the synthetic magnitudes and function dust_transmission

        R={"G_N":3.258,
           "R_N":2.176,
           "Z_N":1.199,
           "G_S":3.212,
           "R_S":2.164,
           "Z_S":1.211,
           "G_G":2.197,
           "BP_G":2.844,
           "RP_G":1.622,
        }

    # Add GALEX - see https://github.com/dstndstn/tractor/issues/99
    R.update({
        'FUV_N': 6.793,
        'NUV_N': 6.620,
        'FUV_S': 6.793,
        'NUV_S': 6.620})

    # Add WISE from
    # https://github.com/dstndstn/tractor/blob/main/tractor/sfd.py#L23-L35
    R.update({
        'W1_N': 0.184,
        'W2_N': 0.113,
        'W3_N': 0.0241,
        'W4_N': 0.00910,
        'W1_S': 0.184,
        'W2_S': 0.113,
        'W3_S': 0.0241,
        'W4_S': 0.00910
        })

    assert(band.upper() in ['FUV', 'NUV', "G","R","Z","BP","RP",'W1','W2','W3','W4'])
    assert(photsys.upper() in ["N","S","G"])
    return R["{}_{}".format(band.upper(),photsys.upper())]

def mwdust_transmission(ebv, band, photsys, match_legacy_surveys=False):
    """Convert SFD E(B-V) value to dust transmission 0-1 for band and photsys

    Args:
        ebv (float or array-like): SFD E(B-V) value(s)
        band (str): 'G', 'R', 'Z', 'W1', 'W2', 'W3', or 'W4'
        photsys (str or array of str): 'N' or 'S' imaging surveys photo system

    Returns:
        scalar or array (same as ebv input), Milky Way dust transmission 0-1

    If `photsys` is an array, `ebv` must also be array of same length.
    However, `ebv` can be an array with a str `photsys`.

    Also see `dust_transmission` which returns transmission vs input wavelength
    """
    if isinstance(photsys, str):
        r_band = extinction_total_to_selective_ratio(band, photsys, match_legacy_surveys=match_legacy_surveys)
        a_band = r_band * ebv
        transmission = 10**(-a_band / 2.5)
        return transmission
    else:
        photsys = np.asarray(photsys)
        if np.isscalar(ebv):
            raise ValueError('array photsys requires array ebv')
        if len(ebv) != len(photsys):
            raise ValueError('len(ebv) {} != len(photsys) {}'.format(
                len(ebv), len(photsys)))

        transmission = np.zeros(len(ebv))
        for p in np.unique(photsys):
            ii = (photsys == p)
            r_band = extinction_total_to_selective_ratio(band, p, match_legacy_surveys=match_legacy_surveys)
            a_band = r_band * ebv[ii]
            transmission[ii] = 10**(-a_band / 2.5)

        return transmission

def ext_odonnell(wave, Rv=3.1):
    """Return extinction curve from Odonnell (1994), defined in the wavelength
    range [3030,9091] Angstroms.  Outside this range, use CCM (1989).

    Args:
        wave : 1D array of vacuum wavelength [Angstroms]
        Rv   : Value of R_V (scalar); default is 3.1

    Returns:
        1D array of A(lambda)/A(V)
    """

    # direct python translation of idlutils/pro/dust/ext_odonnell.pro

    A = np.zeros(wave.shape)
    xx = 10000. / wave

    optical_waves = (xx >= 1.1) & (xx <= 3.3)
    other_waves = (xx < 1.1) | (xx > 3.3)

    if np.sum(optical_waves) > 0:
        yy = xx[optical_waves] - 1.82
        afac = (1.0 + 0.104*yy - 0.609*yy**2 + 0.701*yy**3 + 1.137*yy**4 -
                1.718*yy**5 - 0.827*yy**6 + 1.647*yy**7 - 0.505*yy**8)
        bfac = (1.952*yy + 2.908*yy**2 - 3.989*yy**3 - 7.985*yy**4 +
                11.102*yy**5 + 5.491*yy**6 - 10.805*yy**7 + 3.347*yy**8)
        A[optical_waves] = afac + bfac / Rv
    if np.sum(other_waves) > 0:
        A[other_waves] = ext_ccm(wave[other_waves], Rv=Rv)

    return A


def ext_ccm(wave, Rv=3.1):
    """Return extinction curve from CCM (1989), defined in the wavelength
    range [1250,33333] Angstroms.

    Args:
        wave : 1D array of vacuum wavelength [Angstroms]
        Rv   : Value of R_V (scalar); default is 3.1

    Returns:
        1D array of A(lambda)/A(V)
    """

    # direct python translation of idlutils/pro/dust/ext_ccm.pro
    # numeric values checked with other implementation

    A = np.zeros(wave.shape)
    xx = 10000. / wave

    # Limits for CCM fitting function
    qLO = (xx > 8.0)                   # No data, lambda < 1250 Ang
    qUV = (xx > 3.3) & (xx <= 8.0)     # UV + FUV
    qOPT = (xx > 1.1) & (xx <= 3.3)    # Optical/NIR
    qIR = (xx > 0.3) & (xx <= 1.1)     # IR
    qHI = (xx <= 0.3)                  # No data, lambda > 33,333 Ang

    # For lambda < 1250 Ang, arbitrarily return Alam=5
    if np.sum(qLO) > 0:
        A[qLO] = 5.0

    if np.sum(qUV) > 0:
        xt = xx[qUV]
        afac = 1.752 - 0.316*xt - 0.104 / ((xt - 4.67)**2 + 0.341)
        bfac = -3.090 + 1.825*xt + 1.206 / ((xt - 4.62)**2 + 0.263)

        qq = (xt >= 5.9) & (xt <= 8.0)
        if np.sum(qq) > 0:
            Fa = -0.04473*(xt[qq]-5.9)**2 - 0.009779*(xt[qq]-5.9)**3
            Fb = 0.2130*(xt[qq]-5.9)**2 + 0.1207*(xt[qq]-5.9)**3
            afac[qq] += Fa
            bfac[qq] += Fb

        A[qUV] = afac + bfac / Rv

    if np.sum(qOPT) > 0:
        yy = xx[qOPT] - 1.82
        afac = (1.0 + 0.17699*yy - 0.50447*yy**2 - 0.02427*yy**3 +
                0.72085*yy**4 + 0.01979*yy**5 - 0.77530*yy**6 + 0.32999*yy**7)
        bfac = (1.41338*yy + 2.28305*yy**2 + 1.07233*yy**3 -
                5.38434*yy**4 - 0.62251*yy**5 + 5.30260*yy**6 - 2.09002*yy**7)
        A[qOPT] = afac + bfac / Rv

    if np.sum(qIR) > 0:
        yy = xx[qIR]**1.61
        afac = 0.574*yy
        bfac = -0.527*yy
        A[qIR] = afac + bfac / Rv

    # For lambda > 33,333 Ang, arbitrarily extrapolate the IR curve
    if np.sum(qHI) > 0:
        yy = xx[qHI]**1.61
        afac = 0.574*yy
        bfac = -0.527*yy
        A[qHI] = afac + bfac / Rv

    return A

def ext_fitzpatrick(wave, R_V=3.1, avglmc=False, lmc2=False,
                    x0=None,gamma=None,
                    c1=None,c2=None,c3=None,c4=None) :
    """
    Return extinction curve from Fitzpatrick (1999).

    Args:
        wave : 1D array of vacuum wavelength [Angstroms]

    Returns:
        1D array of A(lambda)/A(V)

    OPTIONAL INPUT KEYWORDS
      R_V - scalar specifying the ratio of total to selective extinction
               R(V) = A(V) / E(B - V).  If not specified, then R = 3.1
               Extreme values of R(V) range from 2.3 to 5.3

      avglmc - if set, then the default fit parameters c1,c2,c3,c4,gamma,x0
             are set to the average values determined for reddening in the
             general Large Magellanic Cloud (LMC) field by Misselt et al.
             (1999, ApJ, 515, 128)
      lmc2 - if set, then the fit parameters are set to the values determined
             for the LMC2 field (including 30 Dor) by Misselt et al.
             Note that neither /AVGLMC or /LMC2 will alter the default value
             of R_V which is poorly known for the LMC.

    The following five input keyword parameters allow the user to customize
    the adopted extinction curve.  For example, see Clayton et al. (2003,
    ApJ, 588, 871) for examples of these parameters in different interstellar
    environments.

      x0 - Centroid of 2200 A bump in microns
           (default = 4.596)
      gamma - Width of 2200 A bump in microns
           (default = 0.99)
      c3 - Strength of the 2200 A bump
           (default = 3.23)
      c4 - FUV curvature
           (default = 0.41)
      c2 - Slope of the linear UV extinction component
           (default = -0.824 + 4.717 / R)
      c1 - Intercept of the linear UV extinction component
           (default = 2.030 - 3.007 * c2)

    NOTES:
       (1) The following comparisons between the FM curve and that of Cardelli,
           Clayton, & Mathis (1989), (see ccm_unred.pro):

           (a) - In the UV, the FM and CCM curves are similar for R < 4.0, but
                 diverge for larger R
           (b) - In the optical region, the FM more closely matches the
                 monochromatic extinction, especially near the R band.
       (2)  Many sightlines with peculiar ultraviolet interstellar extinction
               can be represented with the FM curve, if the proper value of
               R(V) is supplied.
    REQUIRED MODULES:
       scipy, numpy
    REVISION HISTORY:
       Written   W. Landsman        Raytheon  STX   October, 1998
       Based on FMRCurve by E. Fitzpatrick (Villanova)
       Added /LMC2 and /AVGLMC keywords,  W. Landsman   August 2000
       Added ExtCurve keyword, J. Wm. Parker   August 2000
       Assume since V5.4 use COMPLEMENT to WHERE  W. Landsman April 2006
       Ported to Python, C. Theissen August 2012
    """

    # copied by J. Guy from E. Schlafly fm_unred function

    from scipy.interpolate import CubicSpline

    x = 10000. / np.array(wave)  # Convert to inverse microns
    curve = np.zeros(x.shape)

    if lmc2 :
        if x0 == None: x0 = 4.626
        if gamma == None: gamma =  1.05
        if c4 == None: c4 = 0.42
        if c3 == None: c3 = 1.92
        if c2 == None: c2 = 1.31
        if c1 == None: c1 = -2.16
    elif avglmc :
        if x0 == None: x0 = 4.596
        if gamma == None: gamma = 0.91
        if c4 == None: c4 = 0.64
        if c3 == None: c3 =  2.73
        if c2 == None: c2 = 1.11
        if c1 == None: c1 = -1.28
    else:
        if x0 == None: x0 = 4.596
        if gamma == None: gamma = 0.99
        if c4 == None: c4 = 0.41
        if c3 == None: c3 =  3.23
        if c2 == None: c2 = -0.824 + 4.717 / R_V
        if c1 == None: c1 = 2.030 - 3.007 * c2

    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and
    # R-dependent coefficients

    xcutuv = 10000.0 / 2700.0
    xspluv = 10000.0 / np.array([2700.0, 2600.0])

    iuv = x >= xcutuv
    iuv_comp = ~iuv

    if len(x[iuv]) > 0: xuv = np.concatenate( (xspluv, x[iuv]) )
    else: xuv = xspluv.copy()

    yuv = c1  + c2 * xuv
    yuv = yuv + c3 * xuv**2 / ( ( xuv**2 - x0**2 )**2 + ( xuv * gamma )**2 )

    filter1 = xuv.copy()
    filter1[xuv <= 5.9] = 5.9

    yuv = yuv + c4 * ( 0.5392 * ( filter1 - 5.9 )**2 + 0.05644 * ( filter1 - 5.9 )**3 )
    yuv = yuv + R_V
    yspluv = yuv[0:2].copy()                  # save spline points

    if len(x[iuv]) > 0: curve[iuv] = yuv[2:len(yuv)]      # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR

    xsplopir = np.concatenate(([0], 10000.0 / np.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0])))
    ysplir   = np.array([0.0, 0.26469, 0.82925]) * R_V / 3.1
    ysplop   = [np.polyval(np.array([2.13572e-04, 1.00270, -4.22809e-01]), R_V ),
                np.polyval(np.array([-7.35778e-05, 1.00216, -5.13540e-02]), R_V ),
                np.polyval(np.array([-3.32598e-05, 1.00184, 7.00127e-01]), R_V ),
                np.polyval(np.array([-4.45636e-05, 7.97809e-04, -5.46959e-03, 1.01707, 1.19456] ), R_V ) ]

    ysplopir = np.concatenate( (ysplir, ysplop) )

    if len(iuv_comp) > 0:
        cubic = CubicSpline(np.concatenate( (xsplopir,xspluv) ),
                            np.concatenate( (ysplopir,yspluv) ), bc_type='natural')
        curve[iuv_comp] = cubic( x[iuv_comp] )

    return curve/R_V

# based on the work from https://ui.adsabs.harvard.edu/abs/2011ApJ...737..103S
# note from Eddie: I recommend applying the SF11 calibration in the following way:
# A(lambda, F99) = A(lambda, F99, rv)/A(1 micron, F99, rv) * SFD_EBV * 1.029.
# That's a definition that only uses monochromatic extinctions,
# so a lot of ambiguity in what extinction means goes away.
# That doesn't look like the 0.86 rescaling that we quote in abstract,
# but it's convenient because it uses only monochromatic extinctions.

def dust_transmission(wave, ebv_sfd, Rv=3.1):
    """
    Return the dust transmission [0-1] vs. wavelength.

    Args:
        wave : 1D array of vacuum wavelength [Angstroms]
        ebv_sfd : E(B-V) as derived from the map of Schlegel, Finkbeiner and Davis (1998)
        Rv : total-to-selective extinction ratio, defaults to 3.1

    Returns:
        1D array of dust transmission (between 0 and 1)

    The routine does internally multiply ebv_sfd by dust.ebv_sfd_scaling.
    The Fitzpatrick dust extinction law is used, given R_V (default 3.1).

    Also see `mwdust_transmission` which return transmission within a filter
    """
    extinction = ext_fitzpatrick(np.atleast_1d(wave),R_V=Rv) / ext_fitzpatrick(np.array([10000.]),R_V=Rv) * ebv_sfd * 1.029
    if np.isscalar(wave) :
        extinction=float(extinction[0])
    return 10**(-extinction/2.5)

# The SFDMap and _Hemisphere classes and the _bilinear_interpolate and ebv
# functions below were copied on Nov/20/2016 from
# https://github.com/kbarbary/sfdmap/ commit: bacdbbd
# which was originally Licensed under an MIT "Expat" license:
#
# Copyright (c) 2016 Kyle Barbary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


def _bilinear_interpolate(data, y, x):
    """Map a two-dimensional integer pixel-array at float coordinates.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Pixelized array of values.
    y : :class:`float` or :class:`~numpy.ndarray`
        y coordinates (each integer y is a row) of
        location in pixel-space at which to interpolate.
    x : :class:`float` or :class:`~numpy.ndarray`
        x coordinates (each integer x is a column) of
        location in pixel-space at which to interpolate.

    Returns
    -------
    :class:`float` or :class:`~numpy.ndarray`
        Interpolated data values at the passed locations.

    Notes
    -----
    Taken in full from https://github.com/kbarbary/sfdmap/
    """
    yfloor = np.floor(y)
    xfloor = np.floor(x)
    yw = y - yfloor
    xw = x - xfloor

    # pixel locations
    y0 = yfloor.astype(np.int32)
    y1 = y0 + 1
    x0 = xfloor.astype(np.int32)
    x1 = x0 + 1

    # clip locations out of range
    ny, nx = data.shape
    y0 = np.maximum(y0, 0)
    y1 = np.minimum(y1, ny-1)
    x0 = np.maximum(x0, 0)
    x1 = np.minimum(x1, nx-1)

    return ((1.0 - xw) * (1.0 - yw) * data[y0, x0] +
            xw * (1.0-yw) * data[y0, x1] +
            (1.0 - xw) * yw * data[y1, x0] +
            xw * yw * data[y1, x1])


class _Hemisphere(object):
    """Represents one of the hemispheres (in a single file).

    Parameters
    ----------
    fname : :class:`str`
        File name containing one hemisphere of the dust map.
    scaling : :class:`float`
        Multiplicative factor by which to scale the dust map.

    Attributes
    ----------
    data : :class:`~numpy.ndarray`
        Pixelated array of dust map values.
    crpix1, crpix2 : :class:`float`
        World Coordinate System: Represent the 1-indexed
        X and Y pixel numbers of the poles.
    lam_scal : :class:`int`
        Number of pixels from b=0 to b=90 deg.
    lam_nsgp : :class:`int`
        +1 for the northern hemisphere, -1 for the south.

    Notes
    -----
    Taken in full from https://github.com/kbarbary/sfdmap/
    """
    def __init__(self, fname, scaling):
        self.data, header = getdata(fname, header=True)
        self.data *= scaling
        self.crpix1 = header['CRPIX1']
        self.crpix2 = header['CRPIX2']
        self.lam_scal = header['LAM_SCAL']
        self.sign = header['LAM_NSGP']  # north = 1, south = -1

    def ebv(self, l, b, interpolate):
        """Project Galactic longitude/latitude to lambert pixels (See SFD98).

        Parameters
        ----------
        l, b : :class:`numpy.ndarray`
            Galactic longitude and latitude.
        interpolate : :class:`bool`
            If ``True`` use bilinear interpolation to obtain values.

        Returns
        -------
        :class:`~numpy.ndarray`
            Reddening values.
        """
        x = (self.crpix1 - 1.0 +
             self.lam_scal * np.cos(l) *
             np.sqrt(1.0 - self.sign * np.sin(b)))
        y = (self.crpix2 - 1.0 -
             self.sign * self.lam_scal * np.sin(l) *
             np.sqrt(1.0 - self.sign * np.sin(b)))

        # Get map values at these pixel coordinates.
        if interpolate:
            return _bilinear_interpolate(self.data, y, x)
        else:
            x = np.round(x).astype(np.int32)
            y = np.round(y).astype(np.int32)

            # some valid coordinates are right on the border (e.g., x/y = 4096)
            x = np.clip(x, 0, self.data.shape[1]-1)
            y = np.clip(y, 0, self.data.shape[0]-1)
            return self.data[y, x]


class SFDMap(object):
    """Map of E(B-V) from Schlegel, Finkbeiner and Davis (1998).

    Use this class for repeated retrieval of E(B-V) values when
    there is no way to retrieve all the values at the same time: It keeps
    a reference to the FITS data from the maps so that each FITS image
    is read only once.

    Parameters
    ----------
    mapdir : :class:`str`, optional, defaults to :envvar:`DUST_DIR`+``/maps``.
        Directory in which to find dust map FITS images, named
        ``SFD_dust_4096_ngp.fits`` and ``SFD_dust_4096_sgp.fits``.
        If not specified, the map directory is derived from the value of
        the :envvar:`DUST_DIR` environment variable, otherwise an empty
        string is used.
    north, south : :class:`str`, optional
        Names of north and south galactic pole FITS files. Defaults are
        ``SFD_dust_4096_ngp.fits`` and ``SFD_dust_4096_sgp.fits``
        respectively.
    scaling : :class:`float`, optional, defaults to 1
        Scale all E(B-V) map values by this multiplicative factor.
        Pass scaling=0.86 for the recalibration from
        `Schlafly & Finkbeiner (2011) <http://adsabs.harvard.edu/abs/2011ApJ...737..103S)>`_.

    Notes
    -----
    Modified from https://github.com/kbarbary/sfdmap/
    """
    def __init__(self, mapdir=None, north="SFD_dust_4096_ngp.fits",
                 south="SFD_dust_4096_sgp.fits", scaling=1.):

        if mapdir is None:
            dustdir = os.environ.get('DUST_DIR')
            if dustdir is None:
                print('Pass mapdir or set $DUST_DIR')
                raise ValueError('Pass mapdir or set $DUST_DIR')
            else:
                mapdir = os.path.join(dustdir, 'maps')

        if not os.path.exists(mapdir):
            print('Dust maps not found in directory {}'.format(mapdir))
            raise ValueError('Dust maps not found in directory {}'.format(mapdir))

        self.mapdir = mapdir

        # don't load maps initially
        self.fnames = {'north': north, 'south': south}
        self.hemispheres = {'north': None, 'south': None}

        self.scaling = scaling

    def ebv(self, *args, **kwargs):
        """Get E(B-V) value(s) at given coordinate(s).

        Parameters
        ----------
        coordinates : :class:`~astropy.coordinates.SkyCoord` or :class:`~numpy.ndarray`
            If one argument is passed, assumed to be an :class:`~astropy.coordinates.SkyCoord`
            instance, in which case the ``frame`` and ``unit`` keyword arguments are
            ignored. If two arguments are passed, they are treated as
            ``latitute, longitude`` (can be scalars or arrays or a tuple), in which
            case the frame and unit are taken from the passed keywords.
        frame : :class:`str`, optional, defaults to ``'icrs'``
            Coordinate frame, if two arguments are passed. Allowed values are any
            :class:`~astropy.coordinates.SkyCoord` frame, and ``'fk5j2000'`` and ``'j2000'``.
        unit : :class:`str`, optional, defaults to ``'degree'``
            Any :class:`~astropy.coordinates.SkyCoord` unit.
        interpolate : :class:`bool`, optional, defaults to ``True``
            Interpolate between the map values using bilinear interpolation.

        Returns
        -------
        :class:`~numpy.ndarray`
            Specific extinction E(B-V) at the given locations.

        Notes
        -----
        Modified from https://github.com/kbarbary/sfdmap/
        """
        # collect kwargs
        frame = kwargs.get('frame', 'icrs')
        unit = kwargs.get('unit', 'degree')
        interpolate = kwargs.get('interpolate', True)

        # ADM convert to a frame understood by SkyCoords
        # ADM (for backwards-compatibility)
        if frame in ('fk5j2000', 'j2000'):
            frame = 'fk5'

        # compatibility: treat single argument 2-tuple as (RA, Dec)
        if (
                (len(args) == 1) and (type(args[0]) is tuple)
                and (len(args[0]) == 2)
        ):
            args = args[0]

        if len(args) == 1:
            # treat object as already an astropy.coordinates.SkyCoords
            try:
                c = args[0]
            except AttributeError:
                raise ValueError("single argument must be "
                                 "astropy.coordinates.SkyCoord")

        elif len(args) == 2:
            lat, lon = args
            c = SkyCoord(lat, lon, unit=unit, frame=frame)

        else:
            raise ValueError("too many arguments")

        # ADM extract Galactic coordinates from astropy
        l, b = c.galactic.l.radian, c.galactic.b.radian

        # Check if l, b are scalar. If so, convert to 1-d arrays.
        # ADM use numpy.atleast_1d. Store whether the
        # ADM passed values were scalars or not
        return_scalar = not np.atleast_1d(l) is l
        l, b = np.atleast_1d(l), np.atleast_1d(b)

        # Initialize return array
        values = np.empty_like(l)

        # Treat north (b>0) separately from south (b<0).
        for pole, mask in (('north', b >= 0), ('south', b < 0)):
            if not np.any(mask):
                continue

            # Initialize hemisphere if it hasn't already been done.
            if self.hemispheres[pole] is None:
                fname = os.path.join(self.mapdir, self.fnames[pole])
                self.hemispheres[pole] = _Hemisphere(fname, self.scaling)

            values[mask] = self.hemispheres[pole].ebv(l[mask], b[mask],
                                                      interpolate)

        if return_scalar:
            return values[0]
        else:
            return values

    def __repr__(self):
        return ("SFDMap(mapdir={!r}, north={!r}, south={!r}, scaling={!r})"
                .format(self.mapdir, self.fnames['north'],
                        self.fnames['south'], self.scaling))


def ebv(*args, **kwargs):
    """Convenience function, equivalent to ``SFDMap().ebv(*args)``.
    """

    m = SFDMap(mapdir=kwargs.get('mapdir', None),
               north=kwargs.get('north', "SFD_dust_4096_ngp.fits"),
               south=kwargs.get('south', "SFD_dust_4096_sgp.fits"),
               scaling=kwargs.get('scaling', 1.))
    return m.ebv(*args, **kwargs)


def gaia_extinction(g, bp, rp, ebv_sfd):
    # return extinction of gaia magnitudes based on Babusiaux2018 (eqn1/tab1)
    # we assume the EBV the original SFD scale
    # we return A_G, A_BP, A_RP
    gaia_poly_coeff = {'G':[0.9761, -0.1704,
                           0.0086, 0.0011, -0.0438, 0.0013, 0.0099],
                      'BP': [1.1517, -0.0871, -0.0333, 0.0173,
                             -0.0230, 0.0006, 0.0043],
                      'RP':[0.6104, -0.0170, -0.0026,
                            -0.0017, -0.0078, 0.00005, 0.0006]}
    ebv = 0.86 * ebv_sfd # Apply Schlafly+11 correction

    gaia_a0 = 3.1 * ebv
    # here I apply a second-order correction for extinction
    # i.e. I use corrected colors after 1 iteration to determine
    # the best final correction
    inmag = {'G':g, 'RP':rp, 'BP':bp}
    retmag= {'BP':bp * 1, 'RP':rp * 1, 'G': g * 1}
    for i in range(10):
        bprp = retmag['BP'] - retmag['RP']
        for band in ['G','BP','RP']:
            curp = gaia_poly_coeff[band]
            dmag = (np.poly1d(gaia_poly_coeff[band][:4][::-1])(bprp) +
                 curp[4] * gaia_a0 + curp[5]*gaia_a0**2 + curp[6]*bprp*gaia_a0
                 )*gaia_a0
            retmag[band] = inmag[band] - dmag
    return {'G':inmag['G']- retmag['G'],
            'BP':inmag['BP']-retmag['BP'],
            'RP':inmag['RP'] - retmag["RP"]}


#-------------------------------------------------------------------------
#- Used for dust development debugging, but not a core part of the
#- desiutil.dust module for end-users

def _get_ext_coeff(temp, photsys, band, ebv_sfd, rv=3.1):
    """
    Obtain extinction coeffiecient A_X/E(B-V)_SFD for black body spectrum
    with a given temperature observed with photsys and band and
    extinction ebv_sfd

    Args:
        temp (float): temperature in Kelvin
        photsys (str): N, S, or G (gaia)
        band (str): g,r,z (if photsys=N or S); G, BP, or RP if photsys=G
        ebv_sfd (float): E(B-V) from SFD dust map
        Rv (float) : Value of extinction law R_V; default is 3.1

    Returns extinction coefficient A_X / E(B-V)_SFD

    Note: this is currently a _hidden function due to its speclite dependency,
    but it could be promoted if needed by external libraries.
    """
    wave = np.linspace(2900, 11000, 4000)
    sed = 1. / (wave/6000)**5 / (np.exp(143877687. / wave / temp) - 1)
    # code to use Munari library
    # http://cdsarc.u-strasbg.fr/ftp/J/A+A/442/1127/disp10A/fluxed_spectra/T_07000/
    # wave = np.loadtxt('LAMBDA_D10.DAT.txt')
    # wave = np.r_[wave, 10999]
    # sed = np.loadtxt('T07000G40M10V000K2SNWNVD10F.ASC')
    # sed = np.r_[sed, sed[-1]]

    trans = dust_transmission(wave, ebv_sfd)

    #- import speclite only if needed
    import speclite.filters

    if photsys == "N" :
        if band.upper() in ["G","R"] :
            filtername = "BASS-{}".format(band.lower())
        else :
            filtername = "MzLS-z"
    elif photsys == "S" :
        filtername = "decam2014-{}".format(band.lower())
    elif photsys == "G":
        filtername = "gaiadr2-{}".format(band)
    else:
        raise Exception('unrecognized photsys')

    filter_response = speclite.filters.load_filter(filtername)

    fluxunits = 1e-17 * u.erg / u.s / u.cm**2 / u.Angstrom
    mag_no_extinction = filter_response.get_ab_magnitude(sed * fluxunits,
                                                           wave)
    mag_with_extinction = filter_response.get_ab_magnitude(sed * trans * fluxunits, wave)

    # f = np.interp(wave, filter_response.wavelength,filter_response.response)
    # fwave = np.sum(sed * f * wave)/np.sum(sed * f)
    Rbis = (mag_with_extinction-mag_no_extinction)/ebv_sfd
    return Rbis

def _main():
    #- Wrapper for development debugging of extinction coeffs
    import argparse
    import matplotlib.pyplot as plt

    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="Runs and displays a comparison between tabulated values of total to selective extinction ratios and values obtained with synthetic mags.")
    args = parser.parse_args()

    plt.figure("reddening")

    wave=np.linspace(3000-1,11000,1000)
    rv=3.1
    ebv_sfd = 1.
    trans = dust_transmission(wave,ebv_sfd)
    ext   = -2.5*np.log10(trans)
    plt.plot(wave,ext,label="-2.5*log10(dust_transmission)")

    # load filters and compute extinctions here
    try :
        import speclite.filters

        temp = 7000
        sed=1./(wave/6000)**5/(np.exp( 143877687./wave/temp)-1) # dummy star with a spectrum close to a 7000K star (for wavelength>4000A)

        photsys_survey={"N":"BASS+MZLS","S":"DECALS", "G": "Gaia"}
        count=0

        photsys_bands = list(itertools.product('NS','GRZ')) + list(itertools.product('G',['G','BP','RP']))
        for photsys,band in photsys_bands:
            count += 1
            color="C{}".format(count%10)

            if photsys=="N" :
                if band.upper() in ["G","R"] :
                    filtername="BASS-{}".format(band.lower())
                else :
                    filtername="MzLS-z"
            elif photsys=="S" :
                filtername="decam2014-{}".format(band.lower())
            elif photsys == "G":
                filtername="gaiadr2-{}".format(band)

            R=extinction_total_to_selective_ratio(band, photsys)

            Rbis = _get_ext_coeff(temp, photsys, band, ebv_sfd, rv=3.1)

            filter_response=speclite.filters.load_filter(filtername)

            f=np.interp(wave,filter_response.wavelength,filter_response.response)
            fwave = np.sum(sed*f*wave)/np.sum(sed*f)

            if count==1 :
                label1="with extinction_total_to_selective_ratio"
            else :
                label1=None

            plt.plot(fwave,R,"x",color=color,label=label1)

            f=np.interp(wave,filter_response.wavelength,filter_response.response)
            ii=(f>0.01)
            plt.plot(wave[ii],f[ii],"--",color=color,label=filtername)
            print("R_{}({}:{})= {:4.3f} =? {:4.3f}, delta = {:5.4f}".format(band,photsys,photsys_survey[photsys],R,Rbis,R-Rbis))


    except ImportError as e :
        print(e)
        print("Cannot import speclite, so no broadband comparison")

    plt.legend(title="For Rv={}".format(rv))
    plt.xlabel("wavelength (A)")
    plt.ylabel("A(wavelength)/E(B-V)")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    _main()
