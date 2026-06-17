#!/usr/bin/env python
"""
groups-to-refit-v1.4.py

Combined script that:
  1. Restores v1.4 groups to v1.3 group definitions where valid
  2. Identifies all groups needing refit (new, mult changed, moved, overlay,
     truncated, or predetermined)
  3. Writes rem-ellipse-v1.3.txt  (ellipse-only removal, wildcard pattern)
     and rem-coadds-v1.3.txt      (full directory removal)
  4. Writes viewer-truncated.fits, viewer-large-refit-v1.4.fits,
     and viewer-ellipse-refit-v1.4.fits

Restoration criteria (groups skipped if any fail):
  - No objects dropped from v1.3 group (drops.csv)
  - No objects newly added in v1.4 (adds.csv)
  - No member ellipse truncated by v1.3 mosaic boundary

Refit sources (all full footprint):
  1. New GROUP_NAMEs in v1.4 not in v1.3
  2. GROUP_MULT changed for shared group names
  3. Object moved groups between v1.3 and v1.4
  4. Object in updates.csv or flags.csv changes between v1.3 and v1.4
  5. Truncation: member ellipse exceeds v1.3 mosaic boundary
  6. Predetermined refit list (SGA2025-v1.3-refit.fits, optional)
"""

import os
import math
import argparse
import numpy as np
import fitsio
from importlib import resources
from astropy.table import Table
from SGA.SGA import get_galaxy_galaxydir
from SGA.coadds import REGIONBITS
from SGA.util import match
from SGA.parent import load_overlays, restore_large_groups
from SGA.qa import to_skyviewer_table
from SGA.logger import log

DEG2RAD = math.pi / 180.0
ARCMIN_PER_DEG = 60.0


# ============================================================================
# Step 2: Truncation check over all objects in restored p_new
# ============================================================================

def find_truncated_objects(p_new, p12_sgaid_props,
                           trunc_margin_arcsec=1.0, debug=False):
    """
    Check every object in p_new that has a base-version mosaic against that
    mosaic's boundary using the square-mosaic projection.

    Returns
    -------
    truncated_cat : Table
        Rows from p_new for truncated objects
    truncated_groups : set of str
        GROUP_NAMEs containing at least one truncated object
    """
    p_gname   = np.asarray(p_new['GROUP_NAME']).astype(str)
    p_objname = np.asarray(p_new['OBJNAME']).astype(str)
    p_sgaid   = np.asarray(p_new['SGAID'])
    p_ra      = np.asarray(p_new['RA'],   dtype=float)
    p_dec     = np.asarray(p_new['DEC'],  dtype=float)
    p_diam    = np.asarray(p_new['DIAM'], dtype=float)
    p_ba      = np.asarray(p_new['BA'],   dtype=float) if 'BA' in p_new.colnames \
        else np.ones(len(p_new))
    p_pa      = np.asarray(p_new['PA'],   dtype=float) if 'PA' in p_new.colnames \
        else np.zeros(len(p_new))

    truncated_idx    = []
    truncated_groups = set()

    for i in range(len(p_new)):
        sgaid = int(p_sgaid[i])
        props = p12_sgaid_props.get(sgaid)
        if props is None:
            continue  # no base-version mosaic — skip

        a_arc  = 0.5 * p_diam[i]
        ba     = p_ba[i]
        ba_eff = ba if (math.isfinite(ba) and ba > 0) else 1.0
        pa_rad = p_pa[i] * DEG2RAD if math.isfinite(p_pa[i]) else 0.0
        cosd   = math.cos(props['dec'] * DEG2RAD)
        dx     = ((p_ra[i] - props['ra'] + 180.0) % 360.0 - 180.0) \
            * cosd * ARCMIN_PER_DEG
        dy     = (p_dec[i] - props['dec']) * ARCMIN_PER_DEG
        amp_x  = math.hypot(math.sin(pa_rad), ba_eff * math.cos(pa_rad))
        amp_y  = math.hypot(math.cos(pa_rad), ba_eff * math.sin(pa_rad))
        x_max  = abs(dx) + a_arc * amp_x
        y_max  = abs(dy) + a_arc * amp_y
        extent = max(x_max, y_max) * 60.0
        r_eff  = props['r_mosaic'] + trunc_margin_arcsec

        if debug:
            log.info(f"    DEBUG {p_objname[i]}: extent={extent:.2f}\" "
                     f"r_mosaic={props['r_mosaic']:.2f}\"")

        if extent > r_eff:
            truncated_idx.append(i)
            truncated_groups.add(str(p_gname[i]))

    log.info(f"Truncated objects (full footprint): {len(truncated_idx):,d} "
             f"in {len(truncated_groups):,d} groups")

    if truncated_idx:
        return p_new[np.array(truncated_idx)], truncated_groups
    return p_new[[]], truncated_groups


# ============================================================================
# Step 3: Groups-to-refit detection
# ============================================================================

def find_refit_groups(p_new, p_base, ov_base, ov_new,
                      refit_predetermined, truncated_groups):
    """
    Identify all groups needing refit after restoration.

    All sources run over the full footprint.

    Returns
    -------
    all_refit : set
    refit_existing : set
        Subset with existing base-version outputs (needs rem file entry)
    new_groups : set
        Truly new groups (no base-version output to delete)
    source_sets : dict
        Per-source group sets for diagnostic reporting
    """
    p_new_names  = np.asarray(p_new['GROUP_NAME']).astype(str)
    p_base_names = np.asarray(p_base['GROUP_NAME']).astype(str)

    def mult_map(names, mults):
        uniq, idx = np.unique(names, return_index=True)
        return dict(zip(uniq, np.asarray(mults)[idx]))

    mult_base = mult_map(p_base_names, p_base['GROUP_MULT'])
    mult_new  = mult_map(p_new_names,  p_new['GROUP_MULT'])

    # Source 1: new groups
    new_groups = set(mult_new) - set(mult_base)
    log.info(f"Source 1 — new groups in new version:        {len(new_groups):,d}")

    # Source 2: GROUP_MULT changed
    mult_changed = {g for g in set(mult_new) & set(mult_base)
                    if mult_new[g] != mult_base[g]}
    log.info(f"Source 2 — GROUP_MULT changed:               {len(mult_changed):,d}")

    # Source 3: objects that moved groups
    mi_new, mi_base = match(p_new['SGAID'], p_base['SGAID'])
    moved_mask  = p_new_names[mi_new] != p_base_names[mi_base]
    moved_old   = set(p_base_names[mi_base][moved_mask])
    moved_new   = set(p_new_names[mi_new][moved_mask])
    moved       = moved_old | moved_new
    log.info(f"Source 3 — objects that moved groups:        "
             f"{np.sum(moved_mask):,d} objects, {len(moved):,d} groups")

    # Source 4: overlay changes
    updated_base = set(np.unique(ov_base.updates['OBJNAME']))
    updated_new  = set(np.unique(ov_new.updates['OBJNAME']))
    flagged_base = set(ov_base.flags['value'][ov_base.flags['target'] == 'OBJNAME'])
    flagged_new  = set(ov_new.flags['value'][ov_new.flags['target'] == 'OBJNAME'])
    overlay_changed = (updated_new - updated_base) | (flagged_base ^ flagged_new)
    base_nmap = {str(n): str(g) for n, g in zip(p_base['OBJNAME'], p_base_names)}
    new_nmap  = {str(n): str(g) for n, g in zip(p_new['OBJNAME'],  p_new_names)}
    overlay   = set()
    for obj in overlay_changed:
        if base_nmap.get(obj): overlay.add(base_nmap[obj])
        if new_nmap.get(obj):  overlay.add(new_nmap[obj])
    log.info(f"Source 4 — overlay-changed groups:           {len(overlay):,d}")

    # Source 5: truncation (full footprint, passed in)
    log.info(f"Source 5 — truncated groups:                 {len(truncated_groups):,d}")

    # Source 6: predetermined (optional)
    predet = set()
    if refit_predetermined is not None and len(refit_predetermined) > 0:
        base_obj_to_group = {str(n): str(g)
                             for n, g in zip(p_base['OBJNAME'], p_base_names)}
        for objname in np.asarray(refit_predetermined['OBJNAME']).astype(str):
            g = base_obj_to_group.get(objname)
            if g:
                predet.add(g)
    log.info(f"Source 6 — predetermined refit:              {len(predet):,d}")

    all_refit      = new_groups | mult_changed | moved | overlay | truncated_groups | predet
    refit_existing = all_refit - new_groups
    log.info(f"Total groups to refit:                       {len(all_refit):,d}")
    log.info(f"  (existing, need rem files):                {len(refit_existing):,d}")

    source_sets = dict(new=new_groups, mult=mult_changed, moved=moved,
                       overlay=overlay, trunc=truncated_groups, predet=predet)
    return all_refit, refit_existing, new_groups, source_sets


# ============================================================================
# Diagnostic: split into coadds (>5 arcmin) and ellipse-only (all), deduped
# ============================================================================

def report_large_refit_groups(p_new, p_base, all_refit, source_sets, mosaic_diam_map=None):
    p_new_names  = np.asarray(p_new['GROUP_NAME']).astype(str)
    p_base_names = np.asarray(p_base['GROUP_NAME']).astype(str)
    p_new_gdiam  = np.asarray(p_new['GROUP_DIAMETER'], dtype=float)
    p_new_gmult  = np.asarray(p_new['GROUP_MULT'])
    p_base_gmult = np.asarray(p_base['GROUP_MULT'])
    p_new_prim   = np.asarray(p_new['GROUP_PRIMARY'], dtype=bool)
    p_new_obj    = np.asarray(p_new['OBJNAME']).astype(str)
    p_new_ra     = np.asarray(p_new['RA'],  dtype=float)
    p_new_dec    = np.asarray(p_new['DEC'], dtype=float)

    uniq_new,  fi_new  = np.unique(p_new_names,  return_index=True)
    uniq_base, fi_base = np.unique(p_base_names, return_index=True)
    diam_map  = dict(zip(uniq_new,  p_new_gdiam[fi_new]))
    mult_new  = dict(zip(uniq_new,  p_new_gmult[fi_new]))
    mult_base = dict(zip(uniq_base, p_base_gmult[fi_base]))

    group_info = {}
    for i in np.where(p_new_prim)[0]:
        g = p_new_names[i]
        if g not in group_info:
            group_info[g] = (p_new_obj[i], p_new_ra[i], p_new_dec[i])

    ellipse_only_sources = {'overlay', 'predet'}

    def src_tag(g):
        return '+'.join(k for k, s in source_sets.items() if g in s) or '?'

    def is_ellipse_only(g):
        return set(k for k, s in source_sets.items() if g in s) <= ellipse_only_sources

    sorted_groups = sorted(all_refit, key=lambda g: diam_map.get(g, 0), reverse=True)

    # Coadds: >1 arcmin, deduped by OBJNAME
    large_coadds = [g for g in sorted_groups
                    if diam_map.get(g, 0) > 1.0 and not is_ellipse_only(g)]
    seen_coadds = {}
    for g in large_coadds:
        d, info, obj = diam_map.get(g, 0), group_info.get(g), None
        obj = info[0] if info else '?'
        if obj not in seen_coadds or d > seen_coadds[obj][1]:
            seen_coadds[obj] = (g, d, info)
    dedup_coadds = sorted(seen_coadds.items(), key=lambda x: x[1][1], reverse=True)

    # Ellipse-only: all groups present in p_new, deduped by OBJNAME
    ellipse_groups_all = [g for g in sorted_groups
                          if is_ellipse_only(g) and g in group_info]
    seen_ellipse = {}
    for g in ellipse_groups_all:
        d, info = diam_map.get(g, 0), group_info.get(g)
        obj = info[0] if info else '?'
        if obj not in seen_ellipse or d > seen_ellipse[obj][1]:
            seen_ellipse[obj] = (g, d, info)
    dedup_ellipse = sorted(seen_ellipse.items(), key=lambda x: x[1][1], reverse=True)

    regions = ['dr11-south', 'dr11-north']

    def region_of(g):
        """Return region of the primary of group g in p_new."""
        info = group_info.get(g)
        if info is None:
            return 'unknown'
        obj = info[0]
        hits = np.where(p_new_obj == obj)[0]
        if len(hits) == 0:
            return 'unknown'
        reg_bits = int(np.asarray(p_new['REGION'])[hits[0]])
        from SGA.coadds import REGIONBITS
        for reg in regions:
            if reg_bits & REGIONBITS[reg]:
                return reg
        return 'unknown'

    def print_table(dedup, label, total):
        log.info("")
        log.info(f"{label} — total {total} groups, {len(dedup)} shown")
        mosaic_diam_map_ = mosaic_diam_map or {}
        hdr = (f"{'OBJNAME':<30}  {'GROUP_NAME':<12}  {'DIAM(arcmin)':>12}  "
               f"{'DIAM_MOSAIC':>12}  {'RA':>10}  {'DEC':>10}  {'Mbase':>5}  {'Mnew':>5}  SOURCE")
        sep = "-" * 116
        for reg in regions:
            reg_dedup = [(obj, v) for obj, v in dedup if region_of(v[0]) == reg]
            if not reg_dedup:
                continue
            log.info(f"  [{reg}] ({len(reg_dedup)} objects)")
            log.info(hdr)
            log.info(sep)
            for objname, (g, d, info) in reg_dedup:
                ra  = info[1] if info else 0.0
                dec = info[2] if info else 0.0
                mb  = mult_base.get(g, 0)
                mn  = mult_new.get(g, 0)
                md  = mosaic_diam_map_.get(g, 0.0)
                log.info(f"{objname:<30}  {g:<12}  {d:>12.3f}  "
                         f"{md:>12.3f}  {ra:>10.5f}  {dec:>10.5f}  {mb:>5d}  {mn:>5d}  {src_tag(g)}")

    print_table(dedup_coadds,
                "COADDS REFIT — groups > 5 arcmin (rem-coadds-v1.3.txt)",
                len([g for g in all_refit if not is_ellipse_only(g)]))
    print_table(dedup_ellipse,
                "ELLIPSE-ONLY REFIT — all groups (rem-ellipse-v1.3.txt)",
                len(ellipse_groups_all))

    return [obj for obj, _ in dedup_coadds], [obj for obj, _ in dedup_ellipse]


# ============================================================================
# Main
# ============================================================================

def main(base_version='v1.3', new_version='v1.4',
         min_diam=0.0, trunc_margin=10.0, verbose=False, debug=False):

    base_parent = f'SGA2025-beta-parent-{base_version}.fits'
    new_parent  = f'SGA2025-beta-parent-{new_version}.fits'
    refit_file  = f'SGA2025-{base_version}-refit.fits'
    datadir     = f'/pscratch/sd/i/ioannis/SGA2025-{base_version}'

    log.info("Loading catalogs...")
    p_new  = Table(fitsio.read(new_parent))
    p_base = Table(fitsio.read(base_parent))
    log.info(f"  parent-{new_version}:  {len(p_new):,d} objects")
    log.info(f"  parent-{base_version}: {len(p_base):,d} objects")

    refit_pre = None
    if os.path.exists(refit_file):
        refit_pre = Table(fitsio.read(refit_file))
        log.info(f"  predetermined refit ({refit_file}): {len(refit_pre):,d} objects")
    else:
        log.info(f"  {refit_file} not found, skipping predetermined refit")

    log.info("Loading overlays...")
    ov_base = load_overlays(
        resources.files('SGA').joinpath(f'data/SGA2025/overlays/{base_version}'))
    ov_new  = load_overlays(
        resources.files('SGA').joinpath(f'data/SGA2025/overlays/{new_version}'))

    # Step 1: restore
    log.info(f"Step 1: Restoring groups (min_diam={min_diam:.1f}')...")
    p_new, p12_sgaid_props, n_restored = restore_large_groups(
        p_new, p_base, ov_base, ov_new,
        min_diam_arcmin=min_diam,
        trunc_margin_arcsec=trunc_margin,
        verbose=verbose,
        debug=debug,
    )

    # Step 2: truncation check on restored p_new
    log.info("Step 2: Checking for truncated objects (full footprint)...")
    truncated_cat, truncated_groups = find_truncated_objects(
        p_new, p12_sgaid_props,
        trunc_margin_arcsec=trunc_margin,
        debug=debug,
    )
    if len(truncated_cat) > 0:
        view_trunc = to_skyviewer_table(truncated_cat)
        view_trunc.write('viewer-truncated.fits', overwrite=True)
        log.info(f"Wrote viewer-truncated.fits "
                 f"({len(truncated_cat):,d} truncated objects across all groups)")

    # Step 3: find all groups needing refit
    log.info("Step 3: Identifying groups to refit...")
    all_refit, refit_existing, new_groups, source_sets = find_refit_groups(
        p_new, p_base, ov_base, ov_new, refit_pre, truncated_groups)

    # Build GROUP_NAME -> mosaic diameter (arcmin) from base-version props
    p_base_names_arr = np.asarray(p_base['GROUP_NAME']).astype(str)
    p_base_sgaid_arr = np.asarray(p_base['SGAID'])
    mosaic_diam_map  = {
        p_base_names_arr[i]: p12_sgaid_props[int(p_base_sgaid_arr[i])]['r_mosaic'] / 30.0
        for i in range(len(p_base))
        if int(p_base_sgaid_arr[i]) in p12_sgaid_props
    }

    dedup_coadds, dedup_ellipse = report_large_refit_groups(
        p_new, p_base, all_refit, source_sets, mosaic_diam_map)

    # Viewers — split by region
    p_new_obj    = np.asarray(p_new['OBJNAME']).astype(str)
    p_new_region = np.asarray(p_new['REGION'])
    regions      = ['dr11-south', 'dr11-north']

    def write_viewer(objnames, tag, label):
        mask = np.isin(p_new_obj, objnames)
        if not np.any(mask):
            return
        for reg in regions:
            reg_mask = mask & ((p_new_region & REGIONBITS[reg]) != 0)
            if not np.any(reg_mask):
                continue
            check = p_new[reg_mask]
            check = check[np.argsort(np.asarray(check['GROUP_DIAMETER']))[::-1]]
            fname = f'viewer-{tag}-{new_version}-{reg}.fits'
            to_skyviewer_table(check, diamcol='DIAM').write(fname, overwrite=True)
            log.info(f"Wrote {fname} ({np.sum(reg_mask):,d} objects)")

    write_viewer(dedup_coadds,  'large-refit',   'coadds refit')
    write_viewer(dedup_ellipse, 'ellipse-refit', 'ellipse refit')

    # Step 4: write removal files using base-version primaries
    log.info("Step 4: Writing removal files...")

    ellipse_only_sources = {'overlay', 'predet'}

    def is_ellipse_only(g):
        return set(k for k, s in source_sets.items() if g in s) <= ellipse_only_sources

    ellipse_groups = {g for g in refit_existing if is_ellipse_only(g)}
    coadds_groups  = refit_existing - ellipse_groups
    log.info(f"  Ellipse-only refit groups: {len(ellipse_groups):,d}")
    log.info(f"  Full coadds refit groups:  {len(coadds_groups):,d}")

    p_base_names = np.asarray(p_base['GROUP_NAME']).astype(str)
    p_base_prim  = np.asarray(p_base['GROUP_PRIMARY'], dtype=bool)

    def get_primaries(groups):
        return p_base[np.isin(p_base_names, list(groups)) & p_base_prim]

    def collect_dirs(primaries):
        dirs = []
        for reg in ['dr11-south', 'dr11-north']:
            I = (np.asarray(primaries['REGION']) & REGIONBITS[reg]) != 0
            if np.sum(I) > 0:
                _, gdir = get_galaxy_galaxydir(
                    primaries[I], region=reg, datadir=datadir)
                dirs.append(gdir)
        return np.unique(np.hstack(dirs)) if dirs else np.array([])

    def regiondir(gdir):
        for reg in ['dr11-south', 'dr11-north']:
            if f'/{reg}/' in gdir:
                return reg
        return 'dr11-south'

    # rem-ellipse: wildcard removal of ellipse files only
    ellipse_primaries = get_primaries(ellipse_groups)
    log.info(f"  Base-version ellipse-only primaries: {len(ellipse_primaries):,d}")
    ellipse_dirs = collect_dirs(ellipse_primaries)
    rem_ellipse  = f'rem-ellipse-{base_version}.txt'
    if len(ellipse_dirs) > 0:
        with open(rem_ellipse, 'w') as F:
            for gdir in ellipse_dirs:
                parts = gdir.rstrip('/').split('/')
                reg   = regiondir(gdir)
                sub   = parts[-2] if len(parts) >= 2 else ''
                name  = parts[-1]
                F.write(f'\\rm /pscratch/sd/i/ioannis/SGA2025-{base_version}/'
                        f'{reg}/{sub}/{name}/*ellipse*\n')
        log.info(f"Wrote {rem_ellipse} ({len(ellipse_dirs):,d} groups)")
    else:
        log.info(f"No ellipse-only directories to remove.")

    # rem-coadds: full directory removal
    coadds_primaries = get_primaries(coadds_groups)
    log.info(f"  Base-version coadds primaries: {len(coadds_primaries):,d}")
    coadds_dirs = collect_dirs(coadds_primaries)
    rem_coadds  = f'rem-coadds-{base_version}.txt'
    if len(coadds_dirs) > 0:
        with open(rem_coadds, 'w') as F:
            for gdir in coadds_dirs:
                F.write(f'rm -rf {gdir}\n')
        log.info(f"Wrote {rem_coadds} ({len(coadds_dirs):,d} directories)")
    else:
        log.info(f"No coadds directories to remove.")

    p_new.write(f'SGA2025-beta-parent-{new_version}-restored.fits', overwrite=True)
    log.info(f"Wrote SGA2025-beta-parent-{new_version}-restored.fits")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-version', default='v1.3',
                        help='Base (old) parent version (default: v1.3)')
    parser.add_argument('--new-version', default='v1.4',
                        help='New parent version (default: v1.4)')
    parser.add_argument('--min-diam', type=float, default=0.0,
                        help='GROUP_DIAMETER threshold for restoration (arcmin, default 0)')
    parser.add_argument('--trunc-margin', type=float, default=10.0,
                        help='Truncation tolerance in arcsec (default 10)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-group RESTORE/SKIP lines')
    parser.add_argument('--debug', action='store_true',
                        help='Print per-member truncation details')
    args = parser.parse_args()

    main(base_version=args.base_version,
         new_version=args.new_version,
         min_diam=args.min_diam,
         trunc_margin=args.trunc_margin,
         verbose=args.verbose,
         debug=args.debug)
