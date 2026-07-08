"""
SGA.cosmo
=========

Thin fiducial-cosmology wrapper used wherever the pipeline needs an exact
luminosity distance or its inverse (a cosmological redshift equivalent to
a known distance), instead of the low-z linear approximation D=cz/H0.

Swap :data:`COSMO` to a different :mod:`astropy.cosmology` instance to
change the assumed cosmology everywhere at once -- e.g. to a
DESI-preferred cosmology -- without touching any calling code.

Typical workflow:

    from SGA.cosmo import luminosity_distance, redshift_at_luminosity_distance

    dist_mpc = luminosity_distance(z)                  # z -> D_L (Mpc)
    z        = redshift_at_luminosity_distance(dist_mpc)  # D_L (Mpc) -> z

"""
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

# Fiducial cosmology. Flat LCDM, H0=70 km/s/Mpc, Om0=0.3.
COSMO = FlatLambdaCDM(H0=70., Om0=0.3)


def luminosity_distance(z):
    """Exact luminosity distance (Mpc) for redshift(s) `z`, using COSMO."""
    return COSMO.luminosity_distance(np.asarray(z, dtype=np.float64)).to_value(u.Mpc)


def luminosity_distance_derivative(z, dz=1e-5):
    """dD_L/dz (Mpc), via central finite difference -- COSMO's D_L(z) has
    no simple closed form to differentiate analytically. Used to
    propagate a redshift uncertainty to a distance uncertainty (or vice
    versa) around the given `z`.
    """
    z = np.asarray(z, dtype=np.float64)
    return (luminosity_distance(z + dz) - luminosity_distance(z - dz)) / (2. * dz)


def redshift_at_luminosity_distance(dist_mpc, zmin=1e-8, zmax=3.0):
    """Redshift equivalent to a known luminosity distance (Mpc), assuming
    pure Hubble flow (no peculiar velocity) -- the inverse of
    :func:`luminosity_distance`. Root-found per object via astropy's
    ``z_at_value``, so intended for a modest number of objects with an
    independently known distance, not bulk/vectorized use across the
    full sample.

    Parameters
    ----------
    dist_mpc : :class:`float` or array-like
    zmin, zmax : :class:`float`
        Bracketing redshift range passed to ``z_at_value``.

    Returns
    -------
    :class:`~numpy.ndarray`

    """
    dist_mpc = np.atleast_1d(np.asarray(dist_mpc, dtype=np.float64))
    return np.array([
        z_at_value(COSMO.luminosity_distance, d * u.Mpc, zmin=zmin, zmax=zmax).value
        for d in dist_mpc
    ])
