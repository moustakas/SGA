"""
LSLGA.misc
==========

Miscellaneous code.

"""
import numpy as np

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
