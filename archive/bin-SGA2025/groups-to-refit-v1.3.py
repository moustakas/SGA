#!/usr/bin/env python
"""
groups-to-refit-v1.3.py

Combined script that:
  1. Restores v1.3 groups to v1.2 group definitions where valid
  2. Identifies all groups needing refit (new, mult changed, moved, overlay,
     truncated, or predetermined)
  3. Writes rem-v1.2.txt (v1.2 directories to delete before refitting)
  4. Writes viewer-truncated.fits and viewer-large-refit-v1.3.fits

Restoration criteria (groups skipped if any fail):
  - No objects dropped from v1.2 group (drops.csv)
  - No objects newly added in v1.3 (adds.csv)
  - No member ellipse truncated by v1.2 mosaic boundary

Refit sources:
  1. New GROUP_NAMEs in v1.3 not in v1.2 (full footprint)
  2. GROUP_MULT changed for shared group names (full footprint)
  3. Object moved groups between v1.2 and v1.3 (full footprint)
  4. Object in updates.csv or flags.csv changes (full footprint)
  5. Truncation: member ellipse exceeds v1.2 mosaic boundary (full footprint)
  6. Predetermined refit list (SGA2025-v1.2-refit.fits)
"""

import math
import argparse
import numpy as np
import fitsio
from importlib import resources
from astropy.table import Table
from SGA.SGA import get_galaxy_galaxydir, get_radius_mosaic
from SGA.coadds import REGIONBITS
from SGA.util import match
from SGA.parent import load_overlays
from SGA.qa import to_skyviewer_table
from SGA.parent import restore_large_groups
from SGA.logger import log

DEG2RAD = math.pi / 180.0
ARCMIN_PER_DEG = 60.0


# ============================================================================
# Geometry helpers
# ============================================================================

def _angular_sep_arcmin(ra1, dec1, ra2, dec2):
    """Great-circle separation in arcmin."""
    cosd = math.cos(0.5 * (dec1 + dec2) * DEG2RAD)
    dra = (ra1 - ra2 + 180.0) % 360.0 - 180.0
    return math.hypot(dra * cosd, dec1 - dec2) * ARCMIN_PER_DEG


def max_diam_in_mosaic(props, row13):
    """
    Return the maximum DIAM (arcmin) for an ellipse that fits entirely inside
    the existing v1.2 square mosaic, accounting for PA and BA.

    The mosaic is square with half-side R = r_mosaic (arcsec) / 60 arcmin.
    For an ellipse with semi-axes (a, BA*a) and position angle PA:
      x_max = |dx| + a * hypot(sin(PA), BA*cos(PA))
      y_max = |dy| + a * hypot(cos(PA), BA*sin(PA))
    Requiring both <= R gives a_max = min over the two constraints.

    Parameters
    ----------
    props : dict
        v1.2 group properties (keys: r_mosaic, ra, dec)
    row13 : table row
        v1.3 row for this object (needs RA, DEC, PA, BA)

    Returns
    -------
    float
        Maximum DIAM in arcmin
    """
    R = props['r_mosaic'] / 60.0
    cosd = math.cos(props['dec'] * DEG2RAD)
    dx = ((float(row13['RA']) - props['ra'] + 180.0) % 360.0 - 180.0) \
        * cosd * ARCMIN_PER_DEG
    dy = (float(row13['DEC']) - props['dec']) * ARCMIN_PER_DEG
    ba = float(row13['BA'])
    ba_eff = ba if (math.isfinite(ba) and ba > 0) else 1.0
    pa_rad = float(row13['PA']) * DEG2RAD if math.isfinite(float(row13['PA'])) \
        else 0.0
    amp_x = math.hypot(math.sin(pa_rad), ba_eff * math.cos(pa_rad))
    amp_y = math.hypot(math.cos(pa_rad), ba_eff * math.sin(pa_rad))
    a_max = min((R - abs(dx)) / amp_x if amp_x > 0 else R,
                (R - abs(dy)) / amp_y if amp_y > 0 else R)
    return 2.0 * a_max


# ============================================================================
# Step 2: Truncation check over all objects in restored p13
# ============================================================================

def find_truncated_objects(p13, p12_sgaid_props,
                           trunc_margin_arcsec=1.0, debug=False):
    """
    Check every object in p13 that has a v1.2 mosaic against that mosaic's
    boundary using the square-mosaic projection.

    Returns
    -------
    truncated_cat : Table
        Rows from p13 for truncated objects
    truncated_groups : set of str
        GROUP_NAMEs containing at least one truncated object
    """
    p13_gname   = np.asarray(p13['GROUP_NAME']).astype(str)
    p13_objname = np.asarray(p13['OBJNAME']).astype(str)
    p13_sgaid   = np.asarray(p13['SGAID'])
    p13_ra      = np.asarray(p13['RA'],   dtype=float)
    p13_dec     = np.asarray(p13['DEC'],  dtype=float)
    p13_diam    = np.asarray(p13['DIAM'], dtype=float)
    p13_ba      = np.asarray(p13['BA'],   dtype=float) if 'BA' in p13.colnames \
        else np.ones(len(p13))
    p13_pa      = np.asarray(p13['PA'],   dtype=float) if 'PA' in p13.colnames \
        else np.zeros(len(p13))

    truncated_idx    = []
    truncated_groups = set()

    for i in range(len(p13)):
        sgaid = int(p13_sgaid[i])
        props = p12_sgaid_props.get(sgaid)
        if props is None:
            continue  # no v1.2 mosaic — skip

        a_arc  = 0.5 * p13_diam[i]
        ba     = p13_ba[i]
        ba_eff = ba if (math.isfinite(ba) and ba > 0) else 1.0
        pa_rad = p13_pa[i] * DEG2RAD if math.isfinite(p13_pa[i]) else 0.0
        cosd   = math.cos(props['dec'] * DEG2RAD)
        dx     = ((p13_ra[i] - props['ra'] + 180.0) % 360.0 - 180.0) \
            * cosd * ARCMIN_PER_DEG
        dy     = (p13_dec[i] - props['dec']) * ARCMIN_PER_DEG
        amp_x  = math.hypot(math.sin(pa_rad), ba_eff * math.cos(pa_rad))
        amp_y  = math.hypot(math.cos(pa_rad), ba_eff * math.sin(pa_rad))
        x_max  = abs(dx) + a_arc * amp_x
        y_max  = abs(dy) + a_arc * amp_y
        extent = max(x_max, y_max) * 60.0
        r_eff  = props['r_mosaic'] + trunc_margin_arcsec

        if debug:
            log.info(f"    DEBUG {p13_objname[i]}: extent={extent:.2f}\" "
                     f"r_mosaic={props['r_mosaic']:.2f}\"")

        if extent > r_eff:
            truncated_idx.append(i)
            truncated_groups.add(str(p13_gname[i]))

    log.info(f"Truncated objects (full footprint): {len(truncated_idx):,d} "
             f"in {len(truncated_groups):,d} groups")

    if truncated_idx:
        return p13[np.array(truncated_idx)], truncated_groups
    return p13[[]], truncated_groups


# ============================================================================
# Step 3: Groups-to-refit detection
# ============================================================================

def find_refit_groups(p13, p12, ov_12, ov_13,
                      refit_predetermined, truncated_groups):
    """
    Identify all groups needing refit after restoration.

    All sources run over the full footprint.

    Returns
    -------
    all_refit : set
        All group names needing refit
    refit_existing : set
        Subset with existing v1.2 outputs (needs rem-v1.2.txt entry)
    new_groups : set
        Truly new groups (no v1.2 output to delete)
    source_sets : dict
        Per-source group sets for diagnostic reporting
    """
    p13_names   = np.asarray(p13['GROUP_NAME']).astype(str)
    p12_names   = np.asarray(p12['GROUP_NAME']).astype(str)

    def mult_map(names, mults):
        uniq, idx = np.unique(names, return_index=True)
        return dict(zip(uniq, np.asarray(mults)[idx]))

    mult_p12  = mult_map(p12_names,  p12['GROUP_MULT'])
    mult_p13  = mult_map(p13_names,  p13['GROUP_MULT'])

    # Source 1: new groups
    new_groups = set(mult_p13) - set(mult_p12)
    log.info(f"Source 1 — new groups in v1.3:               {len(new_groups):,d}")

    # Source 2: GROUP_MULT changed
    mult_changed = {g for g in set(mult_p13) & set(mult_p12)
                    if mult_p13[g] != mult_p12[g]}
    log.info(f"Source 2 — GROUP_MULT changed:               {len(mult_changed):,d}")

    # Source 3: objects that moved groups
    mi_p13, mi_p12 = match(p13['SGAID'], p12['SGAID'])
    moved_mask  = p13_names[mi_p13] != p12_names[mi_p12]
    moved_old   = set(p12_names[mi_p12][moved_mask])
    moved_new   = set(p13_names[mi_p13][moved_mask])
    moved       = moved_old | moved_new
    log.info(f"Source 3 — objects that moved groups:        "
             f"{np.sum(moved_mask):,d} objects, {len(moved):,d} groups")

    # Source 4: overlay changes
    updated_12 = set(np.unique(ov_12.updates['OBJNAME']))
    updated_13 = set(np.unique(ov_13.updates['OBJNAME']))
    flagged_12 = set(ov_12.flags['value'][ov_12.flags['target'] == 'OBJNAME'])
    flagged_13 = set(ov_13.flags['value'][ov_13.flags['target'] == 'OBJNAME'])
    overlay_changed = (updated_12 ^ updated_13) | (flagged_12 ^ flagged_13)
    p12_nmap = {str(n): str(g) for n, g in zip(p12['OBJNAME'], p12_names)}
    p13_nmap = {str(n): str(g) for n, g in zip(p13['OBJNAME'], p13_names)}
    overlay  = set()
    for obj in overlay_changed:
        if p12_nmap.get(obj): overlay.add(p12_nmap[obj])
        if p13_nmap.get(obj): overlay.add(p13_nmap[obj])
    log.info(f"Source 4 — overlay-changed groups:           {len(overlay):,d}")

    # Source 5: truncation (full footprint, passed in)
    log.info(f"Source 5 — truncated groups:                 {len(truncated_groups):,d}")

    # Source 6: predetermined (keyed by OBJNAME, resolve to GROUP_NAME via p12)
    predet = set()
    if refit_predetermined is not None and len(refit_predetermined) > 0:
        p12_obj_to_group = {str(n): str(g)
                            for n, g in zip(p12['OBJNAME'], p12_names)}
        for objname in np.asarray(refit_predetermined['OBJNAME']).astype(str):
            g = p12_obj_to_group.get(objname)
            if g:
                predet.add(g)
    log.info(f"Source 6 — predetermined refit:              {len(predet):,d}")

    all_refit      = new_groups | mult_changed | moved | overlay | truncated_groups | predet
    refit_existing = all_refit - new_groups
    log.info(f"Total groups to refit:                       {len(all_refit):,d}")
    log.info(f"  (existing, need rem-v1.2.txt):             {len(refit_existing):,d}")

    source_sets = dict(new=new_groups, mult=mult_changed, moved=moved,
                       overlay=overlay, trunc=truncated_groups, predet=predet)
    return all_refit, refit_existing, new_groups, source_sets


# ============================================================================
# Diagnostic: groups > 5 arcmin, deduplicated by OBJNAME
# ============================================================================

def report_large_refit_groups(p13, p12, all_refit, source_sets):
    p13_names   = np.asarray(p13['GROUP_NAME']).astype(str)
    p13_gdiam   = np.asarray(p13['GROUP_DIAMETER'], dtype=float)
    p12_names   = np.asarray(p12['GROUP_NAME']).astype(str)
    p13_gmult   = np.asarray(p13['GROUP_MULT'])
    p12_gmult   = np.asarray(p12['GROUP_MULT'])
    p13_prim    = np.asarray(p13['GROUP_PRIMARY'], dtype=bool)
    p13_objname = np.asarray(p13['OBJNAME']).astype(str)
    p13_ra      = np.asarray(p13['RA'],  dtype=float)
    p13_dec     = np.asarray(p13['DEC'], dtype=float)

    uniq_p13, fi13 = np.unique(p13_names, return_index=True)
    uniq_p12, fi12 = np.unique(p12_names, return_index=True)
    diam_map  = dict(zip(uniq_p13, p13_gdiam[fi13]))
    mult_p13  = dict(zip(uniq_p13, p13_gmult[fi13]))
    mult_p12  = dict(zip(uniq_p12, p12_gmult[fi12]))

    group_info = {}
    for i in np.where(p13_prim)[0]:
        g = p13_names[i]
        if g not in group_info:
            group_info[g] = (p13_objname[i], p13_ra[i], p13_dec[i])

    ellipse_only_sources = {'overlay', 'predet'}

    def src_tag(g):
        return '+'.join(k for k, s in source_sets.items() if g in s) or '?'

    def is_ellipse_only(g):
        return set(k for k, s in source_sets.items() if g in s) <= ellipse_only_sources

    sorted_groups = sorted(all_refit, key=lambda g: diam_map.get(g, 0), reverse=True)

    # Coadds: >5 arcmin, deduped by OBJNAME
    large_coadds = [g for g in sorted_groups
                    if diam_map.get(g, 0) > 5.0 and not is_ellipse_only(g)]
    seen_coadds = {}
    for g in large_coadds:
        d    = diam_map.get(g, 0)
        info = group_info.get(g)
        obj  = info[0] if info else '?'
        if obj not in seen_coadds or d > seen_coadds[obj][1]:
            seen_coadds[obj] = (g, d, info)
    dedup_coadds = sorted(seen_coadds.items(), key=lambda x: x[1][1], reverse=True)

    # Ellipse-only: all groups present in p13 (no diameter cut), deduped by OBJNAME
    ellipse_groups_all = [g for g in sorted_groups
                          if is_ellipse_only(g) and g in group_info]
    seen_ellipse = {}
    for g in ellipse_groups_all:
        d    = diam_map.get(g, 0)
        info = group_info.get(g)
        obj  = info[0] if info else '?'
        if obj not in seen_ellipse or d > seen_ellipse[obj][1]:
            seen_ellipse[obj] = (g, d, info)
    dedup_ellipse = sorted(seen_ellipse.items(), key=lambda x: x[1][1], reverse=True)

    def print_table(dedup, label, total):
        log.info("")
        log.info(f"{label}")
        log.info(f"{'OBJNAME':<30}  {'GROUP_NAME':<12}  {'DIAM(arcmin)':>12}  "
                 f"{'RA':>10}  {'DEC':>10}  {'M12':>4}  {'M13':>4}  SOURCE")
        log.info("-" * 100)
        for objname, (g, d, info) in dedup:
            ra  = info[1] if info else 0.0
            dec = info[2] if info else 0.0
            m12 = mult_p12.get(g, 0)
            m13 = mult_p13.get(g, 0)
            log.info(f"{objname:<30}  {g:<12}  {d:>12.3f}  "
                     f"{ra:>10.5f}  {dec:>10.5f}  {m12:>4d}  {m13:>4d}  {src_tag(g)}")
        log.info(f"({len(dedup)} objects shown of {total} total groups)")

    print_table(dedup_coadds,
                "COADDS REFIT — groups > 5 arcmin (rem-coadds-v1.2.txt)",
                len([g for g in all_refit if not is_ellipse_only(g)]))
    print_table(dedup_ellipse,
                "ELLIPSE-ONLY REFIT — all groups (rem-ellipse-v1.2.txt)",
                len(ellipse_groups_all))

    return [obj for obj, _ in dedup_coadds], [obj for obj, _ in dedup_ellipse]


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-diam', type=float, default=0.0,
                        help='GROUP_DIAMETER threshold for restoration (arcmin, default 0)')
    parser.add_argument('--trunc-margin', type=float, default=10.0,
                        help='Truncation tolerance in arcsec (default 1)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-group RESTORE/SKIP lines')
    parser.add_argument('--debug', action='store_true',
                        help='Print per-member truncation details')
    args = parser.parse_args()

    log.info("Loading catalogs...")
    p13 = Table(fitsio.read('SGA2025-beta-parent-v1.3.fits'))
    p12 = Table(fitsio.read('SGA2025-beta-parent-v1.2.fits'))
    log.info(f"  parent-v1.3: {len(p13):,d} objects")
    log.info(f"  parent-v1.2: {len(p12):,d} objects")

    try:
        refit_pre = Table(fitsio.read('SGA2025-v1.2-refit.fits'))
        log.info(f"  predetermined refit: {len(refit_pre):,d} objects")
    except Exception:
        log.warning("  SGA2025-v1.2-refit.fits not found, skipping")
        refit_pre = None

    log.info("Loading overlays...")
    ov_12 = load_overlays(resources.files('SGA').joinpath('data/SGA2025/overlays/v1.2'))
    ov_13 = load_overlays(resources.files('SGA').joinpath('data/SGA2025/overlays/v1.3'))

    # Step 1: restore
    log.info(f"Step 1: Restoring groups (min_diam={args.min_diam:.1f}')...")
    p13, p12_sgaid_props, n_restored = restore_large_groups(
        p13, p12, ov_12, ov_13,
        min_diam_arcmin=args.min_diam,
        trunc_margin_arcsec=args.trunc_margin,
        verbose=args.verbose,
        debug=args.debug,
    )

    # Step 2: truncation check on restored p13
    log.info("Step 2: Checking for truncated objects (full footprint)...")
    truncated_cat, truncated_groups = find_truncated_objects(
        p13, p12_sgaid_props,
        trunc_margin_arcsec=args.trunc_margin,
        debug=args.debug,
    )
    if len(truncated_cat) > 0:
        view_trunc = to_skyviewer_table(truncated_cat)
        view_trunc.write('viewer-truncated.fits', overwrite=True)
        log.info(f"Wrote viewer-truncated.fits ({len(truncated_cat):,d} truncated objects across all groups)")

    # Step 3: find all groups needing refit
    log.info("Step 3: Identifying groups to refit...")
    all_refit, refit_existing, new_groups, source_sets = find_refit_groups(
        p13, p12, ov_12, ov_13, refit_pre, truncated_groups)

    dedup_coadds, dedup_ellipse = report_large_refit_groups(p13, p12, all_refit, source_sets)

    p13_objname_arr = np.asarray(p13['OBJNAME']).astype(str)

    # viewer-large-refit: coadds groups > 5 arcmin
    mask_coadds = np.isin(p13_objname_arr, dedup_coadds)
    if np.any(mask_coadds):
        check = p13[mask_coadds]
        check = check[np.argsort(np.asarray(check['GROUP_DIAMETER']))[::-1]]
        to_skyviewer_table(check, diamcol='DIAM').write(
            'viewer-large-refit-v1.3.fits', overwrite=True)
        log.info(f"Wrote viewer-large-refit-v1.3.fits ({np.sum(mask_coadds):,d} objects)")

    # viewer-ellipse-refit: all ellipse-only groups
    mask_ellipse = np.isin(p13_objname_arr, dedup_ellipse)
    if np.any(mask_ellipse):
        check = p13[mask_ellipse]
        check = check[np.argsort(np.asarray(check['GROUP_DIAMETER']))[::-1]]
        to_skyviewer_table(check, diamcol='DIAM').write(
            'viewer-ellipse-refit-v1.3.fits', overwrite=True)
        log.info(f"Wrote viewer-ellipse-refit-v1.3.fits ({np.sum(mask_ellipse):,d} objects)")

    # Step 4: write rem-ellipse-v1.2.txt and rem-coadds-v1.2.txt
    # Groups whose only refit sources are overlay and/or predet need
    # ellipse-only removal; everything else needs full coadds removal.
    log.info("Step 4: Writing removal files...")

    ellipse_only_sources = {'overlay', 'predet'}
    ellipse_groups = {g for g in refit_existing
                      if set(k for k, s in source_sets.items() if g in s)
                      <= ellipse_only_sources}
    coadds_groups  = refit_existing - ellipse_groups
    log.info(f"  Ellipse-only refit groups: {len(ellipse_groups):,d}")
    log.info(f"  Full coadds refit groups:  {len(coadds_groups):,d}")

    p12_names_arr = np.asarray(p12['GROUP_NAME']).astype(str)
    p12_prim_arr  = np.asarray(p12['GROUP_PRIMARY'], dtype=bool)

    def get_primaries(groups):
        return p12[np.isin(p12_names_arr, list(groups)) & p12_prim_arr]

    def collect_dirs(primaries):
        dirs = []
        for reg in ['dr11-south', 'dr9-north']:
            I = (np.asarray(primaries['REGION']) & REGIONBITS[reg]) != 0
            if np.sum(I) > 0:
                _, gdir = get_galaxy_galaxydir(
                    primaries[I], region=reg,
                    datadir='/pscratch/sd/i/ioannis/SGA2025-v1.2')
                dirs.append(gdir)
        return np.unique(np.hstack(dirs)) if dirs else np.array([])

    def regiondir(gdir):
        """Extract the region subdirectory (e.g. 'dr11-south') from a path."""
        for reg in ['dr11-south', 'dr9-north']:
            if f'/{reg}/' in gdir:
                return reg
        return 'dr11-south'

    # rem-ellipse-v1.2.txt: wildcard removal of ellipse files only
    ellipse_primaries = get_primaries(ellipse_groups)
    log.info(f"  v1.2 ellipse-only primaries: {len(ellipse_primaries):,d}")
    ellipse_dirs = collect_dirs(ellipse_primaries)
    if len(ellipse_dirs) > 0:
        with open('rem-ellipse-v1.2.txt', 'w') as F:
            for gdir in ellipse_dirs:
                # Extract the 3-char subdirectory from the path
                parts = gdir.rstrip('/').split('/')
                reg   = regiondir(gdir)
                sub   = parts[-2] if len(parts) >= 2 else ''
                name  = parts[-1]
                F.write(f'\\rm /pscratch/sd/i/ioannis/SGA2025-v1.2/'
                        f'{reg}/{sub}/{name}/*ellipse*\n')
        log.info(f"Wrote rem-ellipse-v1.2.txt ({len(ellipse_dirs):,d} groups)")
    else:
        log.info("No ellipse-only directories to remove.")

    # rem-coadds-v1.2.txt: full directory removal
    coadds_primaries = get_primaries(coadds_groups)
    log.info(f"  v1.2 coadds primaries: {len(coadds_primaries):,d}")
    coadds_dirs = collect_dirs(coadds_primaries)
    if len(coadds_dirs) > 0:
        with open('rem-coadds-v1.2.txt', 'w') as F:
            for gdir in coadds_dirs:
                F.write(f'rm -rf {gdir}\n')
        log.info(f"Wrote rem-coadds-v1.2.txt ({len(coadds_dirs):,d} directories)")
    else:
        log.info("No coadds directories to remove.")

    p13.write('SGA2025-beta-parent-v1.3-restored.fits', overwrite=True)
    log.info("Wrote SGA2025-beta-parent-v1.3-restored.fits")
