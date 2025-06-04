"""
Copyright (C) 1999-2025, Michele Cappellari

E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
https://purl.org/cappellari/software

If you have found this software useful for your research, 
I would appreciate an acknowledgement to the use of
"the MGE fitting method and software by Cappellari (2002)".

https://ui.adsabs.harvard.edu/abs/2002MNRAS.333..400C

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.
In particular, redistribution of the code is not allowed.

#####################################################################

Changelog
---------
V1.0.0: Written by Michele Cappellari, Padova, 30 Giugno 1999
V1.0.1: Minor improvements. MC, ESO Garching, 27 September 1999
V1.1.0: Made a more robust automatic level selection.
    MC, Leiden, August 2001
V1.1.1: Added compilation options, MC, Leiden 20 May 2002
V1.1.2: Load proper color table for plot. MC, Leiden, 9 May 2003
V1.2.0: Do not use a widget to display the image. Just resample the
    image if needed. Added /QUIET keyword and (xmed,ymed) output.
    After suggestion by R. McDermid. MC, Leiden, 29 July 2003
V1.2.1: Make the procedure work also with very small images,
    where it does not make sense to extract a subimage.
    MC, Leiden, 1 August 2003
V1.2.2: Added NBLOB keyword. Useful when the galaxy is not the
    largest feature in the image. MC, Leiden, 9 October 2004
V1.2.3: Gives an error message if IMG is not a two-dimensional array.
    MC, Leiden, 11 May 2006
V1.2.4: Included optional output keywords INDEX, X and Y.
    MC, Munich, 14 December 2006
V1.2.5: Included LEVELS input keyword. MC, Oxford, 3 June 2009
V1.2.6: Minor changes to the plotting. MC, Oxford, 14 October 2009
V1.2.7: Perform computations in DOUBLE. MC, Oxford, 25 April 2010
V1.2.8: Added /DEVICE keyword to TVELLIPSE call, due to a change in
    that astrolib routine. MC, Oxford, 9 January 2013
V2.0.0: Translated from IDL into Python. MC, Aspen Airport, 8 February 2014
V2.0.1: Fixed bug in determining largest blob and plotting the ellipse.
    Thanks to Davor Krajnovic for reporting the problems with examples.
    MC, Oxford, 22 February 2014
V2.0.2: Support both Python 2.7 and Python 3. MC, Oxford, 25 May 2014
V2.0.3: Use unravel_index. Do not interpolate image for plotting.
    Use imshow(...origin='lower'). MC, Oxford, 21 September 2014
V2.0.4: Subtract mean before computing second moments to reduce
    rounding errors. MC, Oxford, 17 October 2014
V2.0.5: Only plot mask of selected pixels. MC, Oxford, 4 December 2014
V2.0.6: Corrected documentation of Ang and (X, Y). Clarified printed output.
    Thanks to Shravan Shetty for pointing out the inconsistency.
    MC, Oxford, 8 September 2015
V2.0.7: Fixed compatibility with Numpy 1.10. MC, Oxford, 23 October 2015
V2.0.8: Returns major axis instead of sigma of the pixels distribution.
    Updated documentation. MC, Oxford, 14 April 2016
V2.0.9: Use interpolation='nearest' to avoid crash on MacOS.
    MC, Oxford, 14 June 2016
V2.0.10: Use faster np.percentile instead of deprecated Scipy version.
    MC, Oxford, 17 March 2017
V2.0.11: Included .pa attribute with astronomical PA.
    MC, Oxford, 28 July 2017
V2.0.12: Dropped legacy Python 2.7 support. MC, Oxford, 12 May 2018
V2.0.13: Updated documentation.
    MC, Oxford, 5 July 2021
Vx.x.xx: Additional changes are now documented in the global CHANGELOG.rst
    file of the MgeFit package

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import ndimage

class find_galaxy:
    """
    mge.find_galaxy
    ===============

    Purpose
    -------
    Find the largest region of connected pixels (after smoothing) lying above a
    given intensity level of the image.
    This is useful to automatically identify the location and orientation of a
    galaxy in an image, assuming it is the largest positive fluctuation. The
    conventions used by this routine are the same as for the rest of the
    ``MgeFit`` package.

    This procedure uses the weighted first and second moments of the intensity
    distribution for the computation of the galaxy center and position angle.

    Calling Sequence
    ----------------
    .. code-block:: python

        import mgefit as mge

        f = mge.find_galaxy(img, binning=5, fraction=0.1, level=None,
                            nblob=1, plot=False, quiet=False)
        print(f'Ellipticity: {f.eps:.3f}')

    Parameters
    ----------
    img: array_like with shape (nr, nc)
        The galaxy images as a 2-dim array.

    Other Parameters
    ----------------
    binning: float, optional
        pixel scale for the image smoothing applied before selection.
    fraction: float, optional
        This corresponds (approximately) to the fraction
        ``[0 < fraction < 1]`` of the image pixels that one wants to consider
        to estimate galaxy parameters (default ``fraction = 0.1 = 10%``)
    level: float, optional
        Level above which to select pixels to consider in the
        estimate of the galaxy parameters. This is an alternative
        to the use of the ``fraction`` keyword.
    nblob: int, optional
        If ``nblob=1`` (default) the procedure selects the largest feature
        in the image, if ``nblob=2`` the second largest is selected, and so
        on for increasing values of ``nblob``. This is useful when the
        galaxy is not the largest feature in the image.
    plot: boolean, optional
        set ``plot=True`` to display an image in the current graphic window
        showing the pixels used in the computation of the moments.
    quiet: boolean, optional
        Set ``quiet = True`` to suppress printed output.

    Returns
    -------
    Stored as attributes of the ``mge.find_galaxy`` class:

    .eps:
        The galaxy "average" ellipticity ``eps = 1 - b/a = 1 - q'``.
    .pa:
        Standard astronomical position angle (PA) measured counter-clockwise
        from the image ``Y`` axis (assumed to coincide with North). 
        Note: ``f.pa = 270 - f.theta``.
    .theta:
        Position angle measured clock-wise from the image ``X`` axis.
    .majoraxis:
        Maximum distance of any of the the selected pixels from ``(xmed, ymed)``.
        For a standar galaxy surface brightness this corresponds to the
        major axis of the selected isophote.
    .xpeak:
        First index (row) in ``img``, of the pixel containing the galaxy center.
        To be precise this coordinate represents the first index of the brightest
        pixels within a 40x40 pixels box centered on the galaxy average center.

        IMPORTANT: The brightest galaxy pixel is the element ``img[xpeak, ypeak]``.
        In the plot produced by ``find_galaxy``, the brightest pixel has
        coordinates ``(Ypeak, Xpeak)``, with the two coordinates swapped!
    .ypeak:
        Second index (column) in img, of the brightest galaxy pixel.
    .xmed:
        ``X`` coordinate of luminosity weighted galaxy center.
    .ymed:
        ``Y`` coordinate of luminosity weighted galaxy center.
    .ind:
        Flattened indices of the pixels used in the computation of the
        galaxy parameters. ``img.flat[ind]`` gives the intensity of the
        selected pixels.

    Example
    -------
    The command below locates the position and orientation of a galaxy
    in the image ``img`` and produces a plot showing the obtained results
    and the region of the image used in the computation:

    .. code-block:: python

        f = mge.find_galaxy(img, plot=True)
        print(f'Ellipticity: {f.eps:.3f}')

    The command below only uses 2% of the image pixels to estimate
    the intensity weighted moments and show the results:

    .. code-block:: python

        f = mge.find_galaxy(img, fraction=0.02, plot=True)
        print(f'Coordinates of the peak intensity: {f.xpeak} {f.xpeak}')

    ###########################################################################
    """
    def __init__(self, img, binning=5, fraction=0.1, plot=False,
                 nblob=1, level=None, quiet=False):

        assert img.ndim == 2, "IMG must be a two-dimensional array"

        a = ndimage.median_filter(img, binning)

        if level is None:
            level = np.percentile(a, (1 - fraction)*100)

        mask = a > level
        labels, nb = ndimage.label(mask)   # Get blob indices
        sizes = ndimage.sum(mask, labels, np.arange(nb + 1))
        j = np.argsort(sizes)[-nblob]      # find the nblob-th largest blob
        self.ind = np.flatnonzero(labels == j)

        self.second_moments(img)
        self.pa = np.mod(270 - self.theta, 180)  # astronomical PA

        if not quiet:
            print(' Pixels used:', self.ind.size)
            print(' Peak img[j, k]:', self.xpeak, self.ypeak)
            print(' Mean (j, k): %.2f %.2f' % (self.xmed, self.ymed))
            print(' Theta (deg): %.1f' % self.theta)
            print(' Astro PA (deg): %.1f' % self.pa)
            print(' Eps: %.3f' % self.eps)
            print(' Major axis (pix): %.1f' % self.majoraxis)

        if plot:
            ax = plt.gca()
            ax.imshow(np.log(img.clip(img[self.xpeak, self.ypeak]/1e4)),
                      cmap='hot', origin='lower', interpolation='nearest')
            mask[:] = False
            mask.flat[self.ind] = True
            ax.imshow(mask, cmap='binary', interpolation='nearest',
                      origin='lower', alpha=0.3)
            ax.autoscale(False)  # prevents further scaling after imshow()
            mjr = 1.1*self.majoraxis
            yc, xc = self.xmed, self.ymed
            ellipse = patches.Ellipse(xy=(xc, yc), width=2*mjr, fill=False,
                                      height=2*mjr*(1-self.eps), angle=-self.theta,
                                      color='red', linewidth=3)
            ax.add_artist(ellipse)
            ang = np.array([0, np.pi]) - np.radians(self.theta)
            ax.plot(xc - mjr*np.sin(ang), yc + mjr*np.cos(ang), 'g--',
                    xc + mjr*np.cos(ang), yc + mjr*np.sin(ang), 'g-', linewidth=3)
            ax.set_xlabel("pixels")
            ax.set_ylabel("pixels")

#-------------------------------------------------------------------------

    def second_moments(self, img):
        #
        # Restrict the computation of the first and second moments to
        # the region containing the galaxy, defined by vector IND.

        img1 = img.flat[self.ind]
        s = img.shape
        x, y = np.unravel_index(self.ind, s)

        # Compute coefficients of the moment of inertia tensor.
        #
        i = np.sum(img1)
        self.xmed = np.sum(img1*x)/i
        self.ymed = np.sum(img1*y)/i
        x1 = x - self.xmed    # Subtract mean to reduce numerical errors
        y1 = y - self.ymed
        x2 = np.sum(img1*x1**2)/i
        y2 = np.sum(img1*y1**2)/i
        xy = np.sum(img1*x1*y1)/i

        # Diagonalize the moment of inertia tensor.
        # theta is the angle, measured counter-clockwise,
        # from the image Y axis to the galaxy major axis.
        #
        self.theta = np.degrees(np.arctan2(2*xy, x2 - y2)/2) + 90.
        a = (x2 + y2)/2
        b = np.sqrt(((x2 - y2)/2)**2 + xy**2)
        self.eps = 1. - np.sqrt((a - b)/(a + b))
        self.majoraxis = np.sqrt(np.max(x1**2 + y1**2))

        # If the image has many pixels, then compute the coordinates of the
        # highest pixel value inside a 40x40 pixels region centered on the
        # first intensity moments (Xmed,Ymed), otherwise just return the
        # coordinates of the highest pixel value in the whole image.
        #
        n = 20
        xmed1 = int(round(self.xmed))
        ymed1 = int(round(self.ymed))   # Check if subimage fits...
        if n <= xmed1 <= s[0]-n and n <= ymed1 <= s[1]-n:
            img2 = img[xmed1-n:xmed1+n, ymed1-n:ymed1+n]
            ij = np.unravel_index(np.argmax(img2), img2.shape)
            self.xpeak, self.ypeak = ij + np.array([xmed1, ymed1]) - n
        else:            # ...otherwise use full image
            self.xpeak, self.ypeak = np.unravel_index(np.argmax(img), s)

#-------------------------------------------------------------------------
