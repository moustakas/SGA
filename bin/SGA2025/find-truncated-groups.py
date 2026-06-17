#!/usr/bin/env python
"""
find-truncated-groups.py

Identify truncated galaxies in a completed SGA2025 ellipse catalog by checking
whether each fitted ellipse (D26, PA, BA) exceeds the boundary of its mosaic
(sized from GROUP_DIAMETER and GROUP_MULT using get_radius_mosaic).

Usage
-----
python find-truncated-groups.py [--version v1.4] [--region dr11-south]
                                [--trunc-margin 10] [--min-diam 1.0]
                                [--debug]
"""

import math
import argparse
import numpy as np
import fitsio
from astropy.table import Table
from SGA.SGA import get_radius_mosaic, get_galaxy_galaxydir
from SGA.qa import to_skyviewer_table
from SGA.logger import log

DEG2RAD      = math.pi / 180.0
ARCMIN_PER_DEG = 60.0


def find_truncated(cat, trunc_margin_arcsec=10.0, min_diam_arcmin=1.0,
                   debug=False):
    """
    Check every object in the ellipse catalog against its group mosaic boundary.

    Uses the square-mosaic projection:
      x_max = |dx| + a * hypot(sin(PA), BA*cos(PA))
      y_max = |dy| + a * hypot(cos(PA), BA*sin(PA))
    where (dx, dy) is the offset of the object from the group center and
    a = D26/2 is the semi-major axis.

    Parameters
    ----------
    cat : Table
        Ellipse catalog with GROUP_DIAMETER, GROUP_MULT, GROUP_RA, GROUP_DEC,
        GROUP_NAME, GROUP_PRIMARY, OBJNAME, RA, DEC, D26, PA, BA columns.
    trunc_margin_arcsec : float
        Tolerance added to mosaic radius (arcsec).
    min_diam_arcmin : float
        Only print objects whose group has GROUP_DIAMETER >= this value.
    debug : bool
        Print per-object truncation details.

    Returns
    -------
    truncated_cat : Table
        Rows from cat for truncated objects (all diameters).
    truncated_groups : set of str
        GROUP_NAMEs containing at least one truncated object.
    """
    gname    = np.asarray(cat['GROUP_NAME']).astype(str)
    objname  = np.asarray(cat['OBJNAME']).astype(str)
    ra       = np.asarray(cat['RA'],            dtype=float)
    dec      = np.asarray(cat['DEC'],           dtype=float)
    d26      = np.asarray(cat['D26'],           dtype=float)
    ba_col   = np.asarray(cat['BA'],            dtype=float)
    pa_col   = np.asarray(cat['PA'],            dtype=float)
    gra      = np.asarray(cat['GROUP_RA'],      dtype=float)
    gdec     = np.asarray(cat['GROUP_DEC'],     dtype=float)
    gdiam    = np.asarray(cat['GROUP_DIAMETER'], dtype=float)
    gmult    = np.asarray(cat['GROUP_MULT'])

    # Precompute r_mosaic per unique group (keyed by GROUP_NAME)
    uniq_gnames, first_idx = np.unique(gname, return_index=True)
    r_mosaic_map = {}
    for gi, g in enumerate(uniq_gnames):
        fi   = first_idx[gi]
        diam = float(gdiam[fi])
        mult = int(gmult[fi])
        # Note: q_primary intentionally omitted — mosaics built without BA inflation
        r_mosaic_map[g] = get_radius_mosaic(diam, multiplicity=mult)

    truncated_idx    = []
    truncated_groups = set()

    for i in range(len(cat)):
        g       = gname[i]
        r_mosaic = r_mosaic_map.get(g)
        if r_mosaic is None:
            continue

        a_arc  = 0.5 * d26[i]
        ba     = ba_col[i]
        ba_eff = ba if (math.isfinite(ba) and ba > 0) else 1.0
        pa_rad = pa_col[i] * DEG2RAD if math.isfinite(pa_col[i]) else 0.0
        cosd   = math.cos(gdec[i] * DEG2RAD)
        dx     = ((ra[i] - gra[i] + 180.0) % 360.0 - 180.0) * cosd * ARCMIN_PER_DEG
        dy     = (dec[i] - gdec[i]) * ARCMIN_PER_DEG
        amp_x  = math.hypot(math.sin(pa_rad), ba_eff * math.cos(pa_rad))
        amp_y  = math.hypot(math.cos(pa_rad), ba_eff * math.sin(pa_rad))
        x_max  = abs(dx) + a_arc * amp_x
        y_max  = abs(dy) + a_arc * amp_y
        extent = max(x_max, y_max) * 60.0
        r_eff  = r_mosaic + trunc_margin_arcsec

        if debug:
            log.info(f"  DEBUG {objname[i]}: d26={d26[i]:.3f}', ba={ba_eff:.3f}, "
                     f"dx={dx:.3f}' dy={dy:.3f}', "
                     f"amp_x={amp_x:.4f} amp_y={amp_y:.4f}, "
                     f"extent={extent:.2f}\" r_mosaic={r_mosaic:.2f}\"")

        if extent > r_eff:
            truncated_idx.append(i)
            truncated_groups.add(g)

    log.info(f"Truncated objects: {len(truncated_idx):,d} "
             f"in {len(truncated_groups):,d} groups")

    if not truncated_idx:
        return cat[[]], truncated_groups

    trunc_cat = cat[np.array(truncated_idx)]

    # Diagnostic: print objects in groups >= min_diam_arcmin, sorted by D26 desc
    trunc_gdiam = np.asarray(trunc_cat['GROUP_DIAMETER'], dtype=float)
    trunc_d26   = np.asarray(trunc_cat['D26'],            dtype=float)
    large_mask  = trunc_gdiam >= min_diam_arcmin
    large_cat   = trunc_cat[large_mask]

    if len(large_cat) > 0:
        srt = np.argsort(np.asarray(large_cat['D26'], dtype=float))[::-1]
        large_cat = large_cat[srt]

        lg_gname  = np.asarray(large_cat['GROUP_NAME']).astype(str)
        lg_obj    = np.asarray(large_cat['OBJNAME']).astype(str)
        lg_d26    = np.asarray(large_cat['D26'],            dtype=float)
        lg_gdiam  = np.asarray(large_cat['GROUP_DIAMETER'], dtype=float)
        lg_gmult  = np.asarray(large_cat['GROUP_MULT'])
        lg_ra     = np.asarray(large_cat['RA'],  dtype=float)
        lg_dec    = np.asarray(large_cat['DEC'], dtype=float)
        lg_prim   = np.asarray(large_cat['GROUP_PRIMARY'], dtype=bool)

        log.info("")
        log.info(f"Truncated objects in groups >= {min_diam_arcmin:.1f} arcmin "
                 f"({len(large_cat):,d} objects, sorted by D26):")
        log.info(f"{'OBJNAME':<30}  {'GROUP_NAME':<12}  {'D26(arcmin)':>11}  "
                 f"{'MOSAIC_DIAM':>11}  {'RA':>10}  {'DEC':>10}  {'MULT':>4}  PRIMARY")
        log.info("-" * 106)
        for i in range(len(large_cat)):
            r_m    = r_mosaic_map.get(lg_gname[i], 0.0)
            md     = r_m / 30.0  # arcsec -> arcmin (full mosaic diameter)
            is_pri = lg_prim[i]
            log.info(f"{lg_obj[i]:<30}  {lg_gname[i]:<12}  {lg_d26[i]:>11.3f}  "
                     f"{md:>11.3f}  {lg_ra[i]:>10.5f}  {lg_dec[i]:>10.5f}  "
                     f"{lg_gmult[i]:>4d}  {'*' if is_pri else ''}")
        log.info(f"({len(large_cat):,d} objects shown of {len(truncated_idx):,d} total truncated)")

    return trunc_cat, truncated_groups


def main(version='v1.4', region='dr11-south',
         trunc_margin=10.0, min_diam=1.0, debug=False):

    infile = f'SGA2025-beta-{version}-{region}.fits'
    log.info(f"Reading {infile} [ELLIPSE]...")
    #cat = Table(fitsio.read(infile, 'ELLIPSE', header=False))
    #with fitsio.FITS(infile) as F:
    #    data = Table(F['ELLIPSE'].read())
    cat = Table.read(infile, 'ELLIPSE')

    ## Find all problematic rows
    #bad_rows = []
    #for i, val in enumerate(cat['GROUP_NAME']):
    #    try:
    #        str(val)
    #    except UnicodeDecodeError:
    #        bad_rows.append(i)
    #        print(f"Row {i}: {repr(val)}")
    #print(f"Found {len(bad_rows)} bad rows: {bad_rows[:10]}")

    # Check the dtype and a few values
    print(f"Column dtype: {cat['GROUP_NAME'].dtype}")
    print(f"First 5 values: {cat['GROUP_NAME'][:5]}")
    print(f"Type of first element: {type(cat['GROUP_NAME'][0])}")

    # Try different conversion approaches
    try:
        gname = cat['GROUP_NAME'].astype('U')  # Unicode string
        print("Success with astype('U')")
    except Exception as e:
        print(f"Failed with astype('U'): {e}")

    try:
        gname = np.char.decode(cat['GROUP_NAME'], 'latin-1')
        print("Success with np.char.decode")
    except Exception as e:
        print(f"Failed with np.char.decode: {e}")

    try:
        gname = [str(x) for x in cat['GROUP_NAME']]  # List comprehension
        gname = np.array(gname)
        print("Success with list comprehension")
    except Exception as e:
        print(f"Failed with list comprehension: {e}")

    import pdb ; pdb.set_trace()

    log.info(f"  {len(cat):,d} objects")

    trunc_cat, truncated_groups = find_truncated(
        cat,
        trunc_margin_arcsec=trunc_margin,
        min_diam_arcmin=min_diam,
        debug=debug,
    )

    if len(trunc_cat) > 0:
        view = to_skyviewer_table(trunc_cat, diamcol='D26')
        outfile = f'viewer-truncated-{version}-{region}.fits'
        view.write(outfile, overwrite=True)
        log.info(f"Wrote {outfile} ({len(trunc_cat):,d} objects)")
    else:
        log.info("No truncated objects found.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find truncated galaxies in an SGA2025 ellipse catalog.')
    parser.add_argument('--version', default='v1.4',
                        help='Catalog version (default: v1.4)')
    parser.add_argument('--region', default='dr11-south',
                        help='Region (default: dr11-south)')
    parser.add_argument('--trunc-margin', type=float, default=10.0,
                        help='Truncation tolerance in arcsec (default: 10)')
    parser.add_argument('--min-diam', type=float, default=0.,
                        help='Min GROUP_DIAMETER to print (arcmin, default: 1.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Print per-object truncation details')
    args = parser.parse_args()

    main(version=args.version,
         region=args.region,
         trunc_margin=args.trunc_margin,
         min_diam=args.min_diam,
         debug=args.debug)
