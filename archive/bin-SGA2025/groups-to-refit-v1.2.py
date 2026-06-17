#!/usr/bin/env python

import numpy as np
import fitsio
from astropy.table import Table
from SGA.SGA import get_galaxy_galaxydir
from SGA.coadds import REGIONBITS
from SGA.util import match
from SGA.logger import log


refit = Table(fitsio.read('SGA2025-v0.80-refit.fits'))
oo = Table(fitsio.read('SGA2025-beta-parent-v0.70.fits'))
nn = Table(fitsio.read('SGA2025-beta-parent-v0.80.fits'))

nindx, oindx = match(nn['SGAID'], oo['SGAID'])
nmatch = nn[nindx]
omatch = oo[oindx]

# Objects already flagged for refit
already_refit = np.isin(nmatch['SGAID'], refit['SGAID'])
refit_groups = np.unique(nn['GROUP_NAME'][np.isin(nn['SGAID'], refit['SGAID'])])

# 1. Group membership changed
group_name_changed = nmatch['GROUP_NAME'] != omatch['GROUP_NAME']
group_mult_changed = nmatch['GROUP_MULT'] != omatch['GROUP_MULT']

# 2. Position changed (5 arcsec tolerance)
pos_changed = ~np.isclose(nmatch['RA'], omatch['RA'], atol=5/3600) | ~np.isclose(nmatch['DEC'], omatch['DEC'], atol=5/3600)

# 3. Diameter changed significantly (>30%)
diam_ratio = nmatch['DIAM'] / np.clip(omatch['DIAM'], 0.01, None)
diam_changed = (diam_ratio < 0.7) | (diam_ratio > 1.3)

# 4. Objects dropped from a group (group exists in both versions but lost members)
group_lost_member = (nmatch['GROUP_NAME'] == omatch['GROUP_NAME']) & (nmatch['GROUP_MULT'] < omatch['GROUP_MULT'])

# 5. Objects gained in a group (group exists in both versions but gained members)
group_gained_member = (nmatch['GROUP_NAME'] == omatch['GROUP_NAME']) & (nmatch['GROUP_MULT'] > omatch['GROUP_MULT'])

# 6. Objects in v0.70 that are NOT in v0.80 — find their v0.70 groups
dropped_sgaids = oo['SGAID'][~np.isin(oo['SGAID'], nn['SGAID'])]
dropped_groups_v70 = np.unique(oo['GROUP_NAME'][np.isin(oo['SGAID'], dropped_sgaids)])
# Which of these groups still exist in v0.80 (by name)?
stale_groups = dropped_groups_v70[np.isin(dropped_groups_v70, nn['GROUP_NAME'])]
in_stale_group = np.isin(nmatch['GROUP_NAME'], stale_groups)

# Combine all change flags
any_change = (
    group_name_changed |
    group_mult_changed |
    pos_changed |
    diam_changed |
    group_lost_member |
    group_gained_member |
    in_stale_group
)

# Find all groups that need refitting (any member changed)
changed_groups = np.unique(nmatch['GROUP_NAME'][any_change])

# Combine with refit groups
all_groups_needing_refit = np.unique(np.concatenate([changed_groups, refit_groups]))

# Summary statistics
log.info(f"=== Change Detection v0.70 → v0.80 ===")
log.info(f"Objects in v0.70: {len(oo):,d}")
log.info(f"Objects in v0.80: {len(nn):,d}")
log.info(f"Objects matched: {len(nmatch):,d}")
log.info(f"")
log.info(f"Changes detected:")
log.info(f"  GROUP_NAME changed: {np.sum(group_name_changed):,d}")
log.info(f"  GROUP_MULT changed: {np.sum(group_mult_changed):,d}")
log.info(f"  Position changed (>5\"): {np.sum(pos_changed):,d}")
log.info(f"  Diameter changed (>30%): {np.sum(diam_changed):,d}")
log.info(f"  Group lost member: {np.sum(group_lost_member):,d}")
log.info(f"  Group gained member: {np.sum(group_gained_member):,d}")
log.info(f"  In stale group: {np.sum(in_stale_group):,d}")
log.info(f"")
log.info(f"Objects dropped v0.70→v0.80: {len(dropped_sgaids):,d}")
log.info(f"Groups with dropped members that still exist: {len(stale_groups):,d}")
log.info(f"")
log.info(f"=== Refit Summary ===")
log.info(f"Groups from refit catalog: {len(refit_groups):,d}")
log.info(f"Groups from change detection: {len(changed_groups):,d}")
log.info(f"Total groups needing refit: {len(all_groups_needing_refit):,d}")

# Objects in groups needing refit
in_group_needing_refit = np.isin(nn['GROUP_NAME'], all_groups_needing_refit)
log.info(f"Total objects in groups needing refit: {np.sum(in_group_needing_refit):,d}")

# Select primaries for refitting
newrefit = nn[in_group_needing_refit & nn['GROUP_PRIMARY']]
log.info(f"Primary objects to refit: {len(newrefit):,d} groups")

# Build change tracking table for primaries
changes = Table()
changes['OBJNAME'] = nmatch['OBJNAME']
changes['GROUP_NAME'] = nmatch['GROUP_NAME']
changes['GROUP_NAME_CHANGED'] = group_name_changed
changes['GROUP_MULT_CHANGED'] = group_mult_changed
changes['POS_CHANGED'] = pos_changed
changes['DIAM_CHANGED'] = diam_changed
changes['GROUP_LOST_MEMBER'] = group_lost_member
changes['GROUP_GAINED_MEMBER'] = group_gained_member
changes['IN_STALE_GROUP'] = in_stale_group
changes['IN_REFIT_CATALOG'] = already_refit
changes['ANY_CHANGE'] = any_change | already_refit

# Match to newrefit
m_refit, m_changes = match(newrefit['OBJNAME'], changes['OBJNAME'])
newrefit_changes = changes[m_changes]

# Generate directory removal list
remdir = []
for reg in ['dr11-south', 'dr9-north']:
    I = (newrefit['REGION'] & REGIONBITS[reg]) != 0
    if np.sum(I) > 0:
        _, gdir = get_galaxy_galaxydir(newrefit[I], region=reg, datadir='/pscratch/sd/i/ioannis/SGA2025-v0.80')
        remdir.append(gdir)

if remdir:
    remdir = np.unique(np.hstack(remdir))
    with open('rem-v0.80.txt', 'w') as F:
        for rem in remdir:
            F.write(f'rm -rf {rem}\n')
    log.info(f"Wrote {len(remdir)} directories to rem-v0.80.txt")

# Write out the refit catalog with change flags
if len(newrefit) > 0:
    for col in ['GROUP_NAME_CHANGED', 'GROUP_MULT_CHANGED', 'POS_CHANGED',
                'DIAM_CHANGED', 'GROUP_LOST_MEMBER', 'GROUP_GAINED_MEMBER',
                'IN_STALE_GROUP', 'IN_REFIT_CATALOG']:
        newrefit[col] = newrefit_changes[col]
    newrefit.write('SGA2025-v0.80-groups-to-refit.fits', overwrite=True)
    log.info(f"Wrote SGA2025-v0.80-groups-to-refit.fits")
