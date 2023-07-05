#!/usr/bin/env python

import os, pdb
import numpy as np
import fitsio
from glob import glob
from astropy.table import Table, vstack, join
from desitarget.targetmask import desi_mask, bgs_mask, scnd_mask

pvnames = ['PV_BRIGHT_HIGH', 'PV_BRIGHT_MEDIUM', 'PV_BRIGHT_LOW',
           'PV_DARK_HIGH', 'PV_DARK_MEDIUM', 'PV_DARK_LOW']
cols = ['RA', 'DEC', 'REF_CAT', 'REF_ID', 'DESI_TARGET', 'BGS_TARGET', 'SCND_TARGET', 'TARGETID']

out = []

for survey in ['bright', 'dark']:
    print(f'Working on {survey}')
    tfiles = glob(os.getenv('DESI_TARGET')+f'/catalogs/dr9/1.1.1/targets/main/resolve/{survey}/targets-{survey}-*.fits')
    for tfile in tfiles:#[:2]:
        tt = Table(fitsio.read(tfile, 'TARGETS', columns=cols))

        I = np.where((desi_mask.mask('BGS_ANY') & tt['DESI_TARGET'] != 0) * (tt['REF_CAT'] == 'L3'))[0]
        tbgs = []
        if len(I) > 0:
            tbgs = tt[I]
            tbgs['TARG'] = 'BGS'
        
        J = []
        for pvname in pvnames:
            J.append(np.where(scnd_mask.mask(pvname) & tt['SCND_TARGET'] != 0)[0])
        J = np.hstack(J)
        tpv = []
        if len(J) > 0:
            tpv = tt[J]
            tpv['TARG'] = 'PV'

        _out = []
        if len(tbgs) > 0 and len(tpv) > 0:
            _out = join(tbgs, tpv, keys=cols)
            _, uindx = np.unique(_out['TARGETID'], return_index=True)
            _out = _out[uindx]
            if 'TARG_1' in _out.colnames:
                _out['TARG'] = [t1+'-'+t2 for t1, t2 in zip(_out['TARG_1'].data, _out['TARG_2'].data)]
                _out.remove_columns(['TARG_1', 'TARG_2'])
        elif len(tbgs) > 0 and len(tpv) == 0:
            _out = tbgs
        elif len(tbgs) == 0 and len(tpv) > 0:
            _out = tpv
            
        if len(_out) > 0:
            _out['FILENAME'] = os.path.basename(tfile)
            _out['SURVEY'] = survey
            out.append(_out)

out = vstack(out)
# ignores bright-dark overlap
_, uindx = np.unique(out['TARGETID'], return_index=True)
out = out[uindx]
print(len(out))
out.write('sga-bgspv-targets.fits', overwrite=True)
