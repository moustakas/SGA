"""
SGA.groups
==========

Code to do construct various group catalogs.

"""
import os, time, pdb
import numpy as np
from astropy.table import Table


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


def build_group_catalog(cat, mfac=1.5, dmax=3.0/60.0):
    """dmax in degrees

    Group SGA galaxies together where their circular radii would overlap.  Use
    the catalog D25 diameters (in arcmin) multiplied by a scaling factor MFAC.
    The output catalog adds the column GROUP_ID which is unique for each group.
    The column MULT_GROUP is the multiplicity of that galaxy's group.

    """
    from astropy.table import Column
    from pydl.pydlutils.spheregroup import spheregroup
    from astrometry.util.starutil_numpy import degrees_between

    print('Starting spheregrouping.')

    nchar = np.max([len(gg) for gg in cat['SGANAME']])+6 # add six characters for "_GROUP"
    
    t0 = time.time()
    cat.add_column(Column(name='GROUP_ID', data=np.zeros(len(cat), dtype=np.int32)-1))
    cat.add_column(Column(name='GROUP_NAME', length=len(cat), dtype=f'<U{nchar}'))
    cat.add_column(Column(name='GROUP_MULT', data=np.zeros(len(cat), dtype=np.int16)))
    cat.add_column(Column(name='GROUP_PRIMARY', data=np.zeros(len(cat), dtype=bool)))
    cat.add_column(Column(name='GROUP_RA', length=len(cat), dtype='f8')) # diameter-weighted center
    cat.add_column(Column(name='GROUP_DEC', length=len(cat), dtype='f8'))
    cat.add_column(Column(name='GROUP_DIAM', length=len(cat), dtype='f4'))

    # Initialize a unique group number for each galaxy
    gnum = np.arange(len(cat)).astype(np.int32)
    mgrp = np.ones(len(cat)).astype(np.int16)

    ra, dec, diam = cat['RA'].value, cat['DEC'].value, cat['DIAM'].value
    
    # First group galaxies within dmax arcmin, setting those to have the same
    # group number
    t0 = time.time()
    print('Spheregrouping took...', end='')
    ingroup, group_mult, firstgroup, nextgroup = spheregroup(ra, dec, dmax)

    ngroup = np.count_nonzero(firstgroup != -1)
    for ii in np.arange(ngroup):
        #print(ii, ngroup)
        nn = group_mult[ii] # number of galaxies in this group
        if nn > 1:
            # Build INDX as the indices of all objects in this grouping
            indx = np.zeros(nn, dtype=int)
            indx[0] = firstgroup[ii]
            for jj in np.arange(nn-1):
                indx[jj+1] = nextgroup[indx[jj]]
            # Look at all pairs within this grouping to see if they should be connected.
            for jj in np.arange(nn-1):
                for kk in np.arange(jj, nn):
                    dd = degrees_between(ra[indx[jj]], dec[indx[jj]], ra[indx[kk]], dec[indx[kk]])
                    # If these two galaxies should be connected, make GNUM the
                    # same for them...
                    #print(dd, mfac * (cat['DIAM'][indx[jj]] / 60. + cat['DIAM'][indx[kk]] / 60.))
                    if dd < (0.5 * mfac * (diam[indx[jj]] / 60. + diam[indx[kk]] / 60.)):
                        jndx = np.where(np.logical_or(gnum[indx]==gnum[indx[jj]], gnum[indx]==gnum[indx[kk]]))[0]
                        gnum[indx[jndx]] = gnum[indx[jndx[0]]]
                        mgrp[indx[jndx]] = len(jndx)
            #print(ii, ngroup, gnum[indx], mgrp[indx])

    # Special-case the largest galaxies, looking for neighbhors
    ibig = np.where(diam / 60. > dmax)[0]
    if len(ibig) > 0:
        for ii in np.arange(len(ibig)):
           dd = degrees_between(ra[ibig[ii]], dec[ibig[ii]], ra, dec)
           inear = np.where(dd < 0.5*(cat[ibig[ii]]['DIAM_INIT'] + diam) / 60.)[0]
           if len(inear) > 0:
               for jj in np.arange(len(inear)):
                  indx = np.where(np.logical_or(gnum==gnum[ibig[ii]], gnum==gnum[inear[jj]]))[0]
                  gnum[indx] = gnum[indx[0]]
                  mgrp[indx] = len(indx)
    print('...{:.3f} min'.format((time.time() - t0)/60))

    npergrp, _ = np.histogram(gnum, bins=len(gnum), range=(0, len(gnum)))

    print(f'Found {len(set(gnum))} total groups, including:')
    print(f'  {int(np.sum((npergrp == 1)))} groups with 1 member')
    print(f'  {int(np.sum((npergrp == 2)))} groups with 2 members')
    print(f'  {int(np.sum((npergrp > 2) * (npergrp <= 5)))} group(s) with 3-5 members')
    print(f'  {int(np.sum((npergrp > 5) * (npergrp <= 10)))} group(s) with 6-10 members')
    print(f'  {int(np.sum( (npergrp > 10)))} group(s) with >10 members')

    cat['GROUP_ID'] = gnum
    cat['GROUP_MULT'] = mgrp

    I = np.where(cat['GROUP_MULT'] == 1)[0]
    if len(I) > 0:
        cat['GROUP_RA'][I] = ra[I]
        cat['GROUP_DEC'][I] = dec[I]
        cat['GROUP_DIAM'][I] = diam[I]
        cat['GROUP_NAME'][I] = cat['SGANAME'][I]
        cat['GROUP_PRIMARY'][I] = True

    more = np.where(cat['GROUP_MULT'] > 1)[0]
    for group in set(cat['GROUP_ID'][more]):
        I = np.where(cat['GROUP_ID'] == group)[0]
        # Compute the DIAM-weighted RA, Dec of the group:
        weight = diam[I]
        cat['GROUP_RA'][I] = np.sum(weight * ra[I]) / np.sum(weight)
        cat['GROUP_DEC'][I] = np.sum(weight * dec[I]) / np.sum(weight)
        # Get the diameter of the group as the distance between the center of
        # the group and the outermost galaxy (plus the diameter of that galaxy,
        # in case it's a big one!).
        dd = degrees_between(ra[I], dec[I], cat['GROUP_RA'][I[0]], cat['GROUP_DEC'][I[0]])
        pad = dd + diam[I] / 60.0
        gdiam = 2 * np.max(pad) * 60 # [arcmin]
        # cap the maximum size of the group
        if gdiam > 15.:# and len(I) <= 2:
            gdiam = 1.1 * np.max(pad) * 60 # [arcmin]
        cat['GROUP_DIAM'][I] = gdiam
        if cat['GROUP_DIAM'][I[0]] < np.max(diam[I]):
            print('Should not happen!')
            raise ValueError

        # Assign the group name based on its largest member and also make this
        # galaxy "primary".
        primary = np.argmax(diam[I])
        cat['GROUP_NAME'][I] = f'{cat["SGANAME"][I][primary]}_GROUP'
        cat['GROUP_PRIMARY'][I[primary]] = True

        #if cat['GROUP_ID'][I][0] == 2708:
        #    pdb.set_trace()
        
    print(f'Building a group catalog took {(time.time() - t0)/60.:.3f} min')
        
    return cat


def qa(version='v1'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context='talk', style='ticks', font_scale=1.2)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax[0, 0].scatter(ra[m1], dec[m1], s=5)
    ax[0, 0].scatter(ra[miss], dec[miss], s=5)
    ax[0, 0].set_xlim(290, 90)
    ax[0, 0].set_xlabel('RA')
    ax[0, 0].set_ylabel('Dec')

    ax[0, 1].hist(cat['RADIUS'][m1]*2/60, bins=50, range=(0, 8),
                  label='SGA-match (N={})'.format(len(m1)))
    ax[0, 1].hist(cat['RADIUS'][miss]*2/60, bins=50, range=(0, 8), alpha=0.5,
                  label='SGA-no match (N={})'.format(len(miss)))
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_xlabel('log Radius (arcmin)')
    ax[0, 1].set_ylabel('Number of Galaxies')
    ax[0, 1].legend(loc='upper right', fontsize=14)

    ax[1, 0].scatter(sga['DIAM_INIT'][m2], cat['RADIUS'][m1]*2/60, s=5)
    ax[1, 0].set_xlabel('SGA Diameter [arcmin]')
    ax[1, 0].set_ylabel('Input Diameter [arcmin]')

    ax[1, 1].axis('off')

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.98, hspace=0.25, wspace=0.2)
    fig.savefig(os.path.join(homedir, 'qa-virgofilaments-{}-SGA.png'.format(version)))

