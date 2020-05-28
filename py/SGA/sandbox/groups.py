"""
LSLGA.groups
============

Code to do construct various group catalogs.

"""
from __future__ import absolute_import, division, print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style='ticks', font_scale=1.4, palette='Set2')

PIXSCALE = 0.262

def fof_groups(cat, linking_length=2, verbose=True):
    """Find groups using a friends-of-friends algorithm.
    
    """
    from pydl.pydlutils.spheregroup import spheregroup

    grp, mult, frst, nxt = spheregroup(cat['ra'], cat['dec'], linking_length / 60.0)
    ngrp = max(grp) + 1

    if verbose:
        npergrp, _ = np.histogram(grp, bins=len(grp), range=(0, len(grp)))
        print('Found {} total groups, including:'.format(ngrp), flush=True)
        print('  {} groups with 1 member'.format(
            np.sum( (npergrp == 1) ).astype('int')), flush=True)
        print('  {} groups with 2-5 members'.format(
            np.sum( (npergrp > 1)*(npergrp <= 5) ).astype('int')), flush=True)
        print('  {} groups with 5-10 members'.format(
            np.sum( (npergrp > 5)*(npergrp <= 10) ).astype('int')), flush=True)
        print('  {} groups with >10 members'.format(
            np.sum( (npergrp > 10) ).astype('int')), flush=True)
        
    return (grp, mult, frst, nxt)

def build_groupcat_sky(parent, linking_length=2, verbose=True, groupcatfile='groupcat.fits',
                       parentfile='parent.fits'):
    """Build a group catalog based on just RA, Dec coordinates.

    """
    from astropy.table import Column, Table
    from astrometry.util.starutil_numpy import radectoxyz, xyztoradec, arcsec_between

    grp, mult, frst, nxt = fof_groups(parent, linking_length=linking_length, verbose=verbose)

    ngrp = max(grp) + 1    
    groupid = np.arange(ngrp)
    
    groupcat = Table()
    groupcat.add_column(Column(name='groupid', dtype='i4', length=ngrp, data=groupid)) # unique ID number
    #groupcat.add_column(Column(name='galaxy', dtype='S1000', length=ngrp))
    groupcat.add_column(Column(name='nmembers', dtype='i4', length=ngrp))
    groupcat.add_column(Column(name='ra', dtype='f8', length=ngrp))  # average RA
    groupcat.add_column(Column(name='dec', dtype='f8', length=ngrp)) # average Dec
    groupcat.add_column(Column(name='width', dtype='f4', length=ngrp)) # maximum separation
    groupcat.add_column(Column(name='d25max', dtype='f4', length=ngrp))
    groupcat.add_column(Column(name='d25min', dtype='f4', length=ngrp))
    groupcat.add_column(Column(name='fracmasked', dtype='f4', length=ngrp))
    
    # Add the groupid to the input catalog.
    outparent = parent.copy()
    
    #t0 = time.time()
    npergrp, _ = np.histogram(grp, bins=len(grp), range=(0, len(grp)))
    #print('Time to build the histogram = {:.3f} minutes.'.format( (time.time() - t0) / 60 ) )    
    
    big = np.where( npergrp > 1 )[0]
    small = np.where( npergrp == 1 )[0]

    if len(small) > 0:
        groupcat['nmembers'][small] = 1
        groupcat['groupid'][small] = groupid[small]
        groupcat['ra'][small] = parent['ra'][grp[small]]
        groupcat['dec'][small] = parent['dec'][grp[small]]
        groupcat['d25max'][small] = parent['d25'][grp[small]]
        groupcat['d25min'][small] = parent['d25'][grp[small]]
        groupcat['width'][small] = parent['d25'][grp[small]]
        
        outparent['groupid'][grp[small]] = groupid[small]

    for igrp in range(len(big)):
        jj = frst[big[igrp]]
        ig = list()
        ig.append(jj)
        while (nxt[jj] != -1):
            ig.append(nxt[jj])
            jj = nxt[jj]
        ig = np.array(ig)
        
        ra1, dec1 = parent['ra'][ig].data, parent['dec'][ig].data        
        ra2, dec2 = xyztoradec(np.mean(radectoxyz(ra1, dec1), axis=0))

        groupcat['ra'][big[igrp]] = ra2
        groupcat['dec'][big[igrp]] = dec2
        
        d25min, d25max = np.min(parent['d25'][ig]), np.max(parent['d25'][ig])

        groupcat['d25max'][big[igrp]] = d25max
        groupcat['d25min'][big[igrp]] = d25min
        
        groupcat['nmembers'][big[igrp]] = len(ig)
        outparent['groupid'][ig] = groupcat['groupid'][big[igrp]]
        
        # Get the distance of each object from every other object.
        #diff = arcsec_between(ra1, dec1, ra2, dec2) / 60 # [arcmin] # group center
        
        diff = list()
        for _ra, _dec in zip(ra1, dec1):
            diff.append(arcsec_between(ra1, dec1, _ra, _dec) / 60) # [arcmin]
        
        #if len(ig) > 2:
        #    import pdb ; pdb.set_trace()
        diameter = np.hstack(diff).max()
        groupcat['width'][big[igrp]] = diameter
            
    print('Writing {}'.format(groupcatfile))
    groupcat.write(groupcatfile, overwrite=True)    

    print('Writing {}'.format(parentfile))
    outparent.write(parentfile, overwrite=True)
    
    return groupcat, outparent
