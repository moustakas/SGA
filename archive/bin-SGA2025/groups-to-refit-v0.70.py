#!/usr/bin/env python

import numpy as np
import fitsio
from astropy.table import Table
from SGA.SGA import get_galaxy_galaxydir
from SGA.coadds import REGIONBITS
from SGA.util import match
from SGA.logger import log


refit = Table(fitsio.read('SGA2025-v0.70-refit.fits'))
oo = Table(fitsio.read('SGA2025-beta-parent-v0.60.fits'))
nn = Table(fitsio.read('SGA2025-beta-parent-v0.70.fits'))

nindx, oindx = match(nn['SGAID'], oo['SGAID'])
nmatch = nn[nindx]
omatch = oo[oindx]

already_refit = np.isin(nmatch['SGAID'], refit['SGAID'])

# 1. Group membership changed
group_name_changed = nmatch['GROUP_NAME'] != omatch['GROUP_NAME']
group_mult_changed = nmatch['GROUP_MULT'] != omatch['GROUP_MULT']

# 2. Position changed (1 arcsec tolerance)
pos_changed = ~np.isclose(nmatch['RA'], omatch['RA'], atol=1e-3) | ~np.isclose(nmatch['DEC'], omatch['DEC'], atol=1e-3)

# 3. Diameter changed significantly (>10%)
diam_changed = ~np.isclose(nmatch['DIAM'], omatch['DIAM'], rtol=0.2)

# 4. Shape changed significantly
ba_changed = ~np.isclose(nmatch['BA'], omatch['BA'], atol=0.2)
pa_diff = np.abs(((nmatch['PA'] - omatch['PA'] + 90.0) % 180.0) - 90.0)
pa_changed = pa_diff > 30.0  # degrees

# 5. Objects dropped from a group (group exists in both versions but lost members)
# These are insidious: GROUP_NAME unchanged but GROUP_MULT decreased
group_lost_member = (nmatch['GROUP_NAME'] == omatch['GROUP_NAME']) & (nmatch['GROUP_MULT'] < omatch['GROUP_MULT'])

# 6. Objects in v0.60 that are NOT in v0.70 — find their v0.60 groups
dropped_sgaids = oo['SGAID'][~np.isin(oo['SGAID'], nn['SGAID'])]
dropped_groups_v60 = np.unique(oo['GROUP_NAME'][np.isin(oo['SGAID'], dropped_sgaids)])
# Which of these groups still exist in v0.70 (by name)?
stale_groups = dropped_groups_v60[np.isin(dropped_groups_v60, nn['GROUP_NAME'])]
in_stale_group = np.isin(nmatch['GROUP_NAME'], stale_groups)

# Combine all change flags
any_change = (
    group_name_changed |
    group_mult_changed |
    pos_changed |
    diam_changed |
    ba_changed |
    pa_changed |
    group_lost_member |
    in_stale_group
)

# Find all groups that need refitting (any member changed)
changed_groups = np.unique(nmatch['GROUP_NAME'][any_change & ~already_refit])

# Also include groups from already-refit objects (for completeness check)
refit_groups = np.unique(refit['GROUP_NAME']) if 'GROUP_NAME' in refit.colnames else np.array([])

# Final set of groups needing refit (excluding already done)
groups_needing_refit = changed_groups[~np.isin(changed_groups, refit_groups)]

# Objects in those groups
in_group_needing_refit = np.isin(nmatch['GROUP_NAME'], groups_needing_refit)
needs_refit = in_group_needing_refit & ~already_refit

# Summary statistics
log.info(f"=== Change Detection ===")
log.info(f"Objects matched between v0.60 and v0.70: {len(nmatch):,d}")
log.info(f"Already refit: {np.sum(already_refit):,d}")
log.info(f"")
log.info(f"Changes detected (excluding already refit):")
log.info(f"  GROUP_NAME changed: {np.sum(group_name_changed & ~already_refit):,d}")
log.info(f"  GROUP_MULT changed: {np.sum(group_mult_changed & ~already_refit):,d}")
log.info(f"  Position changed (>1\"): {np.sum(pos_changed & ~already_refit):,d}")
log.info(f"  Diameter changed (>10%): {np.sum(diam_changed & ~already_refit):,d}")
log.info(f"  BA changed (>0.05): {np.sum(ba_changed & ~already_refit):,d}")
log.info(f"  PA changed (>10°): {np.sum(pa_changed & ~already_refit):,d}")
log.info(f"  Group lost member: {np.sum(group_lost_member & ~already_refit):,d}")
log.info(f"  In stale group (dropped members): {np.sum(in_stale_group & ~already_refit):,d}")
log.info(f"")
log.info(f"Objects dropped from v0.60→v0.70: {len(dropped_sgaids):,d}")
log.info(f"Groups with dropped members that still exist: {len(stale_groups):,d}")
log.info(f"")
log.info(f"=== Refit Summary ===")
log.info(f"Groups already refit: {len(refit_groups):,d}")
log.info(f"Additional groups needing refit: {len(groups_needing_refit):,d}")
log.info(f"Additional objects needing refit: {np.sum(needs_refit):,d}")

# Select primaries for refitting
newrefit = nmatch[needs_refit & nmatch['GROUP_PRIMARY']]
log.info(f"Primary objects to refit: {len(newrefit):,d} (representing {len(groups_needing_refit):,d} groups)")

#nn[np.isin(nn['GROUP_NAME'], groups_needing_refit)]['OBJNAME', 'GROUP_NAME', 'GROUP_MULT', 'GROUP_RA', 'GROUP_DEC', 'RA', 'DEC', 'DIAM', 'BA', 'PA'][:10]
#oo[np.isin(oo['GROUP_NAME'], groups_needing_refit)]['OBJNAME', 'GROUP_NAME', 'GROUP_MULT', 'GROUP_RA', 'GROUP_DEC', 'RA', 'DEC', 'DIAM', 'BA', 'PA'][:10]

remdir = []
for reg in ['dr11-south', 'dr9-north']:
    I = newrefit['REGION'] & REGIONBITS[reg] != 0
    _, gdir = get_galaxy_galaxydir(newrefit[I], region=reg, datadir='/pscratch/sd/i/ioannis/SGA2025-v0.70')
    remdir.append(gdir)
remdir = np.unique(np.hstack(remdir))

with open('rem.txt', 'w') as F:
    for rem in remdir:
        F.write(f'rm -rf {rem}\n')
