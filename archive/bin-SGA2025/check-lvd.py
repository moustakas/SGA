import numpy as np
import fitsio
from astropy.table import Table

from SGA.qa import to_skyviewer_table
from SGA.SGA import SAMPLE
from SGA.logger import log
from SGA.util import match

vv = Table(fitsio.read('SGA2025-parent-v0.10.fits'))
nn = Table(fitsio.read('SGA2025-beta-parent-v1.0.fits'))

nlvd = nn[(nn['SAMPLE'] & SAMPLE['LVD']) != 0]
n_indx, v_indx = match(nlvd['SGAID'], vv['SGAID'])
nlvd = nlvd[n_indx]
vlvd = vv[v_indx]

# Compute changes
diam_ratio = nlvd['DIAM'] / np.clip(vlvd['DIAM'], 0.01, None)
pos_shift_arcsec = np.hypot(
    (nlvd['RA'] - vlvd['RA']) * np.cos(np.deg2rad(nlvd['DEC'])) * 3600,
    (nlvd['DEC'] - vlvd['DEC']) * 3600
)
ba_diff = np.abs(nlvd['BA'] - vlvd['BA'])
pa_diff = np.abs(((nlvd['PA'] - vlvd['PA'] + 90.0) % 180.0) - 90.0)

# Summary stats
log.info(f"LVD geometry changes (v0.10 → v1.0):")
log.info(f"  Diameter ratio: median={np.median(diam_ratio):.2f}, min={np.min(diam_ratio):.2f}, max={np.max(diam_ratio):.2f}")
log.info(f"  Position shift (arcsec): median={np.median(pos_shift_arcsec):.1f}, max={np.max(pos_shift_arcsec):.1f}")
log.info(f"  BA diff: median={np.median(ba_diff):.2f}, max={np.max(ba_diff):.2f}")
log.info(f"  PA diff: median={np.median(pa_diff):.1f}, max={np.max(pa_diff):.1f}")

# Flag significant changes
sig_diam = (diam_ratio < 0.5) | (diam_ratio > 2.0)
sig_pos = pos_shift_arcsec > 10.0
sig_ba = ba_diff > 0.2
sig_pa = pa_diff > 30.0

any_sig = sig_diam | sig_pos | sig_ba | sig_pa
any_sig = (diam_ratio > 1.1) & (diam_ratio <= 2.0)

log.info(f"\nSignificant changes:")
log.info(f"  Diameter (>2x or <0.5x): {np.sum(sig_diam):,d}")
log.info(f"  Position (>10\"): {np.sum(sig_pos):,d}")
log.info(f"  BA (>0.2): {np.sum(sig_ba):,d}")
log.info(f"  PA (>30°): {np.sum(sig_pa):,d}")
log.info(f"  Any significant change: {np.sum(any_sig):,d}")

# Build comparison table
changes = Table()
changes['OBJNAME'] = nlvd['OBJNAME']
changes['DIAM_OLD'] = vlvd['DIAM']
changes['DIAM_NEW'] = nlvd['DIAM']
changes['DIAM_RATIO'] = diam_ratio
changes['POS_SHIFT'] = pos_shift_arcsec
changes['BA_OLD'] = vlvd['BA']
changes['BA_NEW'] = nlvd['BA']
changes['PA_OLD'] = vlvd['PA']
changes['PA_NEW'] = nlvd['PA']
changes['SIG_DIAM'] = sig_diam
changes['SIG_POS'] = sig_pos
changes['SIG_ANY'] = any_sig

view = to_skyviewer_table(vlvd[any_sig])
print(len(view))
view.write('~/Downloads/view.fits', overwrite=True)

# Show objects with significant changes, sorted by diameter ratio
sig_changes = changes[any_sig]
sig_changes = sig_changes[np.argsort(sig_changes['DIAM_RATIO'])[::-1]]
print("\nLVD sources with significant geometry changes:")
print(sig_changes['OBJNAME', 'DIAM_OLD', 'DIAM_NEW', 'DIAM_RATIO', 'POS_SHIFT'])

# Check individual objects specifically
#check = 'd0926+70' # 'dw1120p1337'
#check = 'd0934+70'
#check = 'M101-DF3'
#check = 'NGC 4631-dw3' # 'dw0140p1556'
#check = 'HSC-10' # 'd1015+69' # 'd0939+71' # 'UGC 6451' # 'Scl-MM-Dw3' # 'NGC 6503'
#check = 'ESO 274-001'
check = 'NGC 4700' # 'NGC 5194' # 'NGC 625' # 'Holm IV'
idx = np.where(nlvd['OBJNAME'] == check)[0]
if len(idx) > 0:
    i = idx[0]
    print(f"\n{check}:")
    print(f"  RA: {vlvd['RA'][i]:.6f} → {nlvd['RA'][i]:.6f}")
    print(f"  DEC: {vlvd['DEC'][i]:.6f} → {nlvd['DEC'][i]:.6f}")
    print(f"  DIAM: {vlvd['DIAM'][i]:.4f} → {nlvd['DIAM'][i]:.4f} (ratio={diam_ratio[i]:.2f})")
    print(f"  BA: {vlvd['BA'][i]:.2f} → {nlvd['BA'][i]:.2f}")
    print(f"  PA: {vlvd['PA'][i]:.1f} → {nlvd['PA'][i]:.1f}")
    print(f"  Pos shift: {pos_shift_arcsec[i]:.1f} arcsec")
    print()
    print(f"{check},RA,{vlvd['RA'][i]:.6f},VI")
    print(f"{check},DEC,{vlvd['DEC'][i]:.6f},VI")
    print(f"{check},DIAM,{vlvd['DIAM'][i]:.4f},VI")
    print(f"{check},BA,{vlvd['BA'][i]:.2f},VI")
    print(f"{check},PA,{vlvd['PA'][i]:.1f},VI")
