#!/usr/bin/env python

import os, math
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from SGA.SGA import read_sample, get_radius_mosaic, sga_dir

basedir = os.path.join(sga_dir(), 'sample')

def find_culprit(brickname, version='v1.5', region='dr11-south', margin=1.0,
                 search_radius=5.0):
    """Find which v1.5 group(s) caused a brick to be flagged has_sga=True.

    Fast: spatially pre-filters groups to a small patch around the brick
    before running the exact bounding-box check.
    """
    # --- Load brick list to get exact corners ---
    csv_file = os.path.join(basedir, f'{region}-bricks-with-sga-{version}.csv')
    bricks_csv = Table.read(csv_file)
    row = bricks_csv[bricks_csv['brickname'] == brickname]
    if len(row) == 0:
        print(f'Brick {brickname} not found in {csv_file}')
        return
    row = row[0]
    bra  = float(row['ra'])
    bdec = float(row['dec'])

    # Load exact corners from the brick FITS file (critical near RA=0 wrap)
    if region == 'dr11-south':
        bricks_file = os.path.join(basedir, 'bricks-exist-ls-dr11-early-v2.fits.gz')
    else:
        bricks_file = os.path.join(basedir, 'survey-bricks-dr9-north.fits.gz')
    import fitsio
    bricks_fits = Table(fitsio.read(bricks_file))
    brow = bricks_fits[np.asarray(bricks_fits['brickname'], dtype=str) == brickname]
    if len(brow) == 0:
        print(f'Brick {brickname} not found in {bricks_file}, falling back to approximation')
        bra1, bra2   = bra  - 0.125, bra  + 0.125
        bdec1, bdec2 = bdec - 0.125, bdec + 0.125
    else:
        brow  = brow[0]
        bra1  = float(brow['ra1'])
        bra2  = float(brow['ra2'])
        bdec1 = float(brow['dec1'])
        bdec2 = float(brow['dec2'])
    print(f'Brick {brickname}: ra={bra:.4f}, dec={bdec:.4f}, has_sga={row["has_sga"]}')
    print(f'  Corners: ra=[{bra1:.4f}, {bra2:.4f}], dec=[{bdec1:.4f}, {bdec2:.4f}]')

    # --- Load sample and pre-filter spatially ---
    # Use great-circle angular separation for the pre-filter to handle RA wrap correctly
    sample, _ = read_sample(region=region, version=version)
    gra   = np.asarray(sample['GROUP_RA'],       dtype=float)
    gdec  = np.asarray(sample['GROUP_DEC'],      dtype=float)
    gdiam = np.asarray(sample['GROUP_DIAMETER'], dtype=float)
    gmult = np.asarray(sample['GROUP_MULT'],     dtype=int)

    # Great-circle separation (small-angle approx is fine at 5-10 deg scale)
    dra_gc  = ((gra - bra + 180) % 360 - 180) * np.cos(np.radians(bdec))
    ddec_gc = gdec - bdec
    sep     = np.hypot(dra_gc, ddec_gc)
    nearby  = sep < search_radius
    print(f'  Pre-filter: {nearby.sum()} groups within {search_radius}° of brick')

    # Also show the 5 closest regardless, for debugging
    closest = np.argsort(sep)[:5]
    print(f'  5 closest groups:')
    for k in closest:
        print(f'    {str(sample["OBJNAME"][k]):30s} ra={gra[k]:.4f} dec={gdec[k]:.4f} '
              f'sep={sep[k]:.3f}° diam={gdiam[k]:.3f}\'')

    DEG2ARCSEC = 3600.0
    hits = []
    for k in np.where(nearby)[0]:
        ra0  = gra[k]
        dec0 = gdec[k]
        r_arcsec = get_radius_mosaic(gdiam[k], multiplicity=int(gmult[k]))
        half = r_arcsec * (1.0 + margin) / DEG2ARCSEC

        dec_lo, dec_hi = dec0 - half, dec0 + half
        if bdec2 <= dec_lo or bdec1 >= dec_hi:
            continue

        pole_limit = 89.75
        if dec_lo <= -pole_limit or dec_hi >= pole_limit:
            ra_lo, ra_hi = 0.0, 360.0
        else:
            cosd_k = math.cos(math.radians(dec0))
            dra_k  = half / cosd_k if cosd_k > 1e-3 else 180.0
            if dra_k >= 180.0:
                ra_lo, ra_hi = 0.0, 360.0
            else:
                ra_lo = (ra0 - dra_k) % 360.0
                ra_hi = (ra0 + dra_k) % 360.0

        if ra_lo <= ra_hi:
            ra_ok = bra2 > ra_lo and bra1 < ra_hi
        else:
            ra_ok = bra2 > ra_lo or bra1 < ra_hi

        if ra_ok:
            hits.append((str(sample['OBJNAME'][k]), ra0, dec0,
                         gdiam[k], int(gmult[k]), r_arcsec, half * 60.0,
                         ra_lo, ra_hi))

    if hits:
        print(f'\n  {"OBJNAME":30s} {"GROUP_RA":>10s} {"GROUP_DEC":>10s} '
              f'{"DIAM\'":>8s} {"MULT":>5s} {"r_arcsec":>10s} {"half\'":>8s} '
              f'{"ra_lo":>8s} {"ra_hi":>8s}')
        for h in hits:
            print(f'  {h[0]:30s} {h[1]:10.4f} {h[2]:10.4f} '
                  f'{h[3]:8.3f} {h[4]:5d} {h[5]:10.1f} {h[6]:8.4f} '
                  f'{h[7]:8.4f} {h[8]:8.4f}')
    else:
        print(f'\n  No culprit found within {search_radius}° — '
              f'try --search-radius with a larger value.')


def main():

    old = Table.read(os.path.join(basedir, 'dr11-south-bricks-with-sga-v1.2.csv'))
    new = Table.read(os.path.join(basedir, 'dr11-south-bricks-with-sga-v1.5.csv'))
    assert np.all(old['brickname'] == new['brickname'])

    sga_old = old['has_sga'] == 'True'
    sga_new = new['has_sga'] == 'True'
    no_sga_old = old['has_sga'] == 'False'
    no_sga_new = new['has_sga'] == 'False'

    agree_yes = sga_old & sga_new
    agree_no  = no_sga_old & no_sga_new
    I = sga_old & no_sga_new
    J = no_sga_old & sga_new

    print(f'v1.2 has SGA and v1.5 has SGA:    {np.sum(agree_yes):,d}')
    print(f'v1.2 no SGA  and v1.5 no SGA:     {np.sum(agree_no):,d}')
    print(f'v1.2 has SGA and v1.5 no SGA:     {np.sum(I):,d}')
    print(f'v1.2 no SGA  and v1.5 has SGA:    {np.sum(J):,d}')
    print()
    print('Bricks newly flagged in v1.5:')
    print(new[J])

    outfile = './dr11-sga-v1.5-bricks-to-rerun.csv'
    out = new[J]
    print(f'Wrote {outfile}')
    out.write(outfile, overwrite=True, format='csv')

    outfile = '~/Downloads/dr11-sga-v1.5-bricks-to-rerun.fits'
    out = out['brickname', 'ra', 'dec']
    out.rename_column('brickname', 'name')
    print(f'Wrote {outfile}')
    out.write(outfile, overwrite=True)

    pngfile = os.path.join(basedir, 'check-bricklist.png')
    fig, ax = plt.subplots()
    ax.scatter(new['ra'], new['dec'], alpha=0.5, s=1,
               label=f'All bricks (N={len(new):,d})')
    ax.scatter(new['ra'][J], new['dec'][J], alpha=1, s=30, marker='x',
               color='k', label=f'v1.2 no SGA, v1.5 SGA (N={np.sum(J):,d})')
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 360)
    fig.savefig(pngfile)
    print(f'Wrote {pngfile}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--culprit', metavar='BRICKNAME',
                   help='Identify which group caused this brick to be flagged in v1.5')
    p.add_argument('--version', default='v1.5')
    p.add_argument('--region',  default='dr11-south')
    p.add_argument('--margin',  type=float, default=1.0)
    p.add_argument('--search-radius', type=float, default=5.0,
                   help='Spatial pre-filter radius in degrees (default 5.0)')
    a = p.parse_args()

    if a.culprit:
        find_culprit(a.culprit, version=a.version, region=a.region,
                     margin=a.margin, search_radius=a.search_radius)
    else:
        main()
