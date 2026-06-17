#!/usr/bin/env python

import pdb
import fitsio
import numpy as np
from astropy.table import Table, vstack
import matplotlib as mpl
from itertools import cycle
from SGA.SGA import read_sample

##region = 2**1 # north
#region = 2**0 # south
#region = 'dr9-north'
region = 'dr11-south'

colors = cycle([mpl.colors.to_hex(col) for col in mpl.colormaps['Set2'].colors])

# NGC 3169 group has two giant members -- don't break apart!
# NGC 3190:[LMD2020] B - another example
# NGC 0672

# break apart!
# NGC 3270
# IC 5181
# NGC 4555
# UGC 04197

check_groups = True # False
check_test_bricks = False # True

cols = ['OBJNAME', 'RA', 'DEC', 'REGION', 'DIAM', 'BA', 'PA', 'GROUP_PRIMARY',
        'GROUP_MULT', 'GROUP_DIAMETER', 'GROUP_ID']
_, cat = read_sample(region=region, test_bricks=check_test_bricks)
cat = cat[cols]

if check_test_bricks:
    #cat = cat[cat['GROUP_MULT'] < 4]
    I = cat['GROUP_PRIMARY']
    gcat = cat[I]


if check_groups:
    # at least 2 group members
    #cat = cat[cat['GROUP_MULT'] > 2]
    cat = cat[cat['GROUP_MULT'] == 2]
    #cat = cat[cat['GROUP_MULT'] > 1]

    #mindiam = 15.
    #maxdiam = 1e3
    #mindiam = 10.
    #maxdiam = 15.
    #mindiam = 7.
    #maxdiam = 10.
    #mindiam = 5.
    #maxdiam = 7.
    #mindiam = 3.
    #maxdiam = 5.
    #mindiam = 0.
    #maxdiam = 3.
    mindiam = 0.9
    maxdiam = 1.
    #mindiam = 3.
    #maxdiam = 1e3

    # primary member is larger than XX arcmin
    I = (cat['GROUP_PRIMARY'] *
         #(cat['GROUP_DIAMETER'] >= mindiam) * (cat['GROUP_DIAMETER'] < maxdiam)
         (cat['DIAM'] >= mindiam) * (cat['DIAM'] < maxdiam)
         )
    gcat = cat[I]

#gcat = gcat[np.argsort(gcat['GROUP_MULT'])[::-1]]
gcat = gcat[np.argsort(gcat['GROUP_DIAMETER'])[::-1]]

# for each group
allout = []
count = 0
for gid, prim in zip(gcat['GROUP_ID'].value, gcat['OBJNAME'].value):
    I = cat['GROUP_ID'] == gid
    out = cat['OBJNAME', 'RA', 'DEC', 'DIAM', 'BA', 'PA'][I]
    #if any(['DESI' in name for name in out['OBJNAME'].value]):
    ##if not all(['DESI' in name for name in out['OBJNAME'].value]):
    #    continue

    out.rename_columns(['OBJNAME', 'DIAM', 'BA', 'PA'],
                       ['name', 'radius', 'abratio', 'posAngle'])
    out['radius'] *= 60. / 2.
    out['color'] = [next(colors)] * len(out)
    #out.write(f'~/Downloads/{prim}-group.fits', overwrite=True)
    #_=[print(gal) for gal in out['name']]
    #print()
    allout.append(out)

    for ii in range(len(out)):
        count += 1
        if count % 10 == 0:
            print(f'Object: {count}')


allout = vstack(allout)

#outfile = f'~/Downloads/group-desi-region{region}.fits'
#outfile = f'~/Downloads/group-region{region}-{mindiam:.3f}-{maxdiam:.3f}.fits'
outfile = f'~/Downloads/group-{region}-test.fits'
allout.write(outfile, overwrite=True)
#_=[print(gal) for gal in allout['name'] if 'DESI' in gal]
_=[print(gal) for gal in allout['name']]
print(len(allout))

pdb.set_trace()
