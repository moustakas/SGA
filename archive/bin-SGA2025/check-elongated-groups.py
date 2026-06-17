#!/usr/bin/env python

import numpy as np
import fitsio
from astropy.table import Table
from SGA.qa import to_skyviewer_table
from SGA.coadds import REGIONBITS
from SGA.SGA import SAMPLE
from SGA.util import match
from SGA.logger import log

#vv = Table(fitsio.read('SGA2025-parent-v0.22.fits'))
nn = Table(fitsio.read('SGA2025-beta-parent-v1.1.fits'))
vv = Table(fitsio.read('SGA2025-parent-v0.10.fits'))
#nn = Table(fitsio.read('SGA2025-beta-parent-v1.0.fits'))

# Focus on non-LVD overlap sources in dr11-south with GROUP_DIAMETER in your uninspected range
I = (
    #(nn['REGION'] & REGIONBITS['dr9-north'] != 0) &
    (nn['REGION'] & REGIONBITS['dr11-south'] != 0) &
    (nn['GROUP_DIAMETER'] >= 2.) &
    #(nn['GROUP_DIAMETER'] >= 0.5) &
    #(nn['GROUP_DIAMETER'] >= 5) &
    #(nn['GROUP_DIAMETER'] < 1e3) &
    (nn['GROUP_DIAMETER'] < 2.5) &
    ((nn['SAMPLE'] & SAMPLE['OVERLAP']) == 0) &
    ((nn['SAMPLE'] & SAMPLE['LVD']) == 0)
)

nn_sub = nn[I]
log.info(f"Candidates to inspect: {len(nn_sub):,d}")

# Match to v0.10
n_indx, v_indx = match(nn_sub['SGAID'], vv['SGAID'])
nn_match = nn_sub[n_indx]
vv_match = vv[v_indx]

# Compute changes
diam_ratio = nn_match['DIAM'] / np.clip(vv_match['DIAM'], 0.01, None)
ba_new = nn_match['BA']
ba_old = vv_match['BA']

pa_new = nn_match['PA']
pa_old = vv_match['PA']

# Flag: diameter grew significantly AND became elongated
grew_and_elongated = (diam_ratio > 1.5) & (ba_new < 0.4)

# Even more suspicious: was rounder before, now elongated
was_rounder = ba_old > 0.5
#became_elongated = grew_and_elongated & was_rounder
became_elongated = (grew_and_elongated & was_rounder) | ((diam_ratio > 1.5) & (ba_new < 0.2))

log.info(f"\nFiltering results:")
log.info(f"  Diameter grew >1.5x: {np.sum(diam_ratio > 1.5):,d}")
log.info(f"  Currently elongated (BA < 0.4): {np.sum(ba_new < 0.4):,d}")
log.info(f"  Grew AND elongated: {np.sum(grew_and_elongated):,d}")
log.info(f"  Was rounder (BA > 0.5), now elongated: {np.sum(became_elongated):,d}")

# Build inspection table for primary flagged objects
inspect = Table()
inspect['OBJNAME'] = nn_match['OBJNAME']
inspect['SGAID'] = nn_match['SGAID']
inspect['GROUP_NAME'] = nn_match['GROUP_NAME']
inspect['GROUP_MULT'] = nn_match['GROUP_MULT']
inspect['GROUP_RA'] = nn_match['GROUP_RA']
inspect['GROUP_DEC'] = nn_match['GROUP_DEC']
inspect['GROUP_DIAM'] = nn_match['GROUP_DIAMETER']
inspect['DIAM_RATIO'] = diam_ratio

inspect['RA'] = nn_match['RA']
inspect['DEC'] = nn_match['DEC']

inspect['DIAM_NEW'] = nn_match['DIAM']
inspect['BA_NEW'] = ba_new
inspect['PA_NEW'] = pa_new

inspect['DIAM'] = vv_match['DIAM']
inspect['BA'] = vv_match['BA']
inspect['PA'] = vv_match['PA']

inspect['FLAGGED_PRIMARY'] = became_elongated

# Prioritize: grew a lot AND became very elongated AND was rounder
priority_score = diam_ratio * (1.0 / np.clip(ba_new, 0.1, 1.0)) * np.where(was_rounder, 2.0, 1.0)
inspect['PRIORITY'] = priority_score

# Get unique problem groups
problem_groups = np.unique(nn_match['GROUP_NAME'][became_elongated])
log.info(f"\nUnique problem groups (from primary flags): {len(problem_groups):,d}")

# Now find ALL members of problem groups and check for significant diameter changes
all_problem_members_idx = np.where(np.isin(nn['GROUP_NAME'], problem_groups))[0]
nn_problem = nn[all_problem_members_idx]

# Match these to v0.10
p_indx, pv_indx = match(nn_problem['SGAID'], vv['SGAID'])
nn_problem_match = nn_problem[p_indx]
vv_problem_match = vv[pv_indx]

# Compute diameter ratio for all problem group members
diam_ratio_all = nn_problem_match['DIAM'] / np.clip(vv_problem_match['DIAM'], 0.01, None)

# Flag members with significant diameter change (grew >1.5x OR shrunk <0.7x)
#sig_diam_change = (diam_ratio_all > 2.)
sig_diam_change = (diam_ratio_all > 1.5) | (diam_ratio_all < 0.7)

log.info(f"All members in problem groups: {len(nn_problem_match):,d}")
log.info(f"Members with significant diameter change: {np.sum(sig_diam_change):,d}")

# Build full inspection table for all affected members
inspect_all = Table()
inspect_all['OBJNAME'] = nn_problem_match['OBJNAME']
inspect_all['SGAID'] = nn_problem_match['SGAID']
inspect_all['GROUP_NAME'] = nn_problem_match['GROUP_NAME']
inspect_all['GROUP_MULT'] = nn_problem_match['GROUP_MULT']
inspect_all['GROUP_RA'] = nn_problem_match['GROUP_RA']
inspect_all['GROUP_DEC'] = nn_problem_match['GROUP_DEC']
inspect_all['GROUP_DIAM'] = nn_problem_match['GROUP_DIAMETER']
inspect_all['DIAM_RATIO'] = diam_ratio_all

inspect_all['RA'] = nn_problem_match['RA']
inspect_all['DEC'] = nn_problem_match['DEC']

inspect_all['DIAM_NEW'] = nn_problem_match['DIAM']
inspect_all['BA_NEW'] = nn_problem_match['BA']
inspect_all['PA_NEW'] = nn_problem_match['PA']

inspect_all['DIAM'] = vv_problem_match['DIAM']
inspect_all['BA'] = vv_problem_match['BA']
inspect_all['PA'] = vv_problem_match['PA']

# Flag type: primary (elongated) vs secondary (significant diam change in same group)
inspect_all['FLAGGED_PRIMARY'] = np.isin(nn_problem_match['SGAID'], nn_match['SGAID'][became_elongated])
inspect_all['SIG_DIAM_CHANGE'] = sig_diam_change

# Priority: primary flags get highest, then by diameter ratio
inspect_all['PRIORITY'] = np.where(inspect_all['FLAGGED_PRIMARY'],
                                    1000 + np.abs(diam_ratio_all - 1),
                                    np.abs(diam_ratio_all - 1))

# Sort by group then priority
inspect_all = inspect_all[np.lexsort((inspect_all['PRIORITY'][::-1], inspect_all['GROUP_NAME']))]

# Summary by group
log.info(f"\nSummary by group:")
for gname in problem_groups[:10]:
    gmembers = inspect_all[inspect_all['GROUP_NAME'] == gname]
    n_primary = np.sum(gmembers['FLAGGED_PRIMARY'])
    n_sig = np.sum(gmembers['SIG_DIAM_CHANGE'])
    log.info(f"  {gname}: {len(gmembers)} members, {n_primary} primary flagged, {n_sig} sig diam change")

# Show example
log.info(f"\nExample group 06505m5144:")
ex = inspect_all[inspect_all['GROUP_NAME'] == '06505m5144']
if len(ex) > 0:
    print(ex['OBJNAME', 'DIAM', 'DIAM_NEW', 'DIAM_RATIO', 'BA', 'BA_NEW', 'FLAGGED_PRIMARY', 'SIG_DIAM_CHANGE'])

# Write out for inspection - include all members with any flag
inspect_out = inspect_all[inspect_all['FLAGGED_PRIMARY'] | inspect_all['SIG_DIAM_CHANGE']]
log.info(f"\nTotal objects to inspect: {len(inspect_out):,d}")
log.info(f"  Primary flagged (elongated): {np.sum(inspect_out['FLAGGED_PRIMARY']):,d}")
log.info(f"  Secondary flagged (sig diam change): {np.sum(~inspect_out['FLAGGED_PRIMARY'] & inspect_out['SIG_DIAM_CHANGE']):,d}")

#inspect_out = inspect_out[np.argsort(inspect_out['DIAM'])]
_ = [print(f'{obj},') for obj in inspect_out['OBJNAME'].value]
view = to_skyviewer_table(inspect_out)
view['color'] = np.where(inspect_out['FLAGGED_PRIMARY'], 'white', 'black')
view.write('~/Downloads/view.fits', overwrite=True)
log.info(f"Wrote {len(view)} objects to ~/Downloads/view.fits (red=primary, yellow=secondary)")

with open('elongated-tofix.csv', 'w') as F:
    for obj, diam, ba, pa in inspect_out['OBJNAME', 'DIAM', 'BA', 'PA'].iterrows():
        F.write(f'{obj},DIAM,{diam:.3f},VI\n')
        F.write(f'{obj},BA,{ba:.3f},VI\n')
        F.write(f'{obj},PA,{pa:.1f},VI\n')
