"""
SGA.util
========

Support utilities.

"""
import pdb
import numpy as np
from astropy.table import Table

from astrometry.util.starutil_numpy import arcsec_between
from astrometry.libkd.spherematch import match_radec


def parse_geometry(cat, ref):
    """Parse a specific set of elliptical geometry.

    ref - choose from among SGA2020, HYPERLEDA, RC3, LIT

    """
    nobj = len(cat)
    diam = np.zeros(nobj) - 99. # [arcsec]
    ba = np.ones(nobj)
    pa = np.zeros(nobj)

    if ref == 'SGA2020':
        I = cat['DIAM_SGA2020'] > 0.
        if np.any(I):
            diam[I] = cat[I]['DIAM_SGA2020'] * 60. # [arcsec]
            ba[I] = cat[I]['BA_SGA2020']
            pa[I] = cat[I]['PA_SGA2020']
    elif ref == 'RC3':
        I = (cat['DIAM_LIT'] > 0.) * (cat['DIAM_LIT_REF'] == 'RC3')
        if np.any(I):
            diam[I] = cat[I]['DIAM_LIT'] * 60. # [arcsec]
            ba[I] = cat[I]['BA_LIT']
            pa[I] = cat[I]['PA_LIT']
    elif ref == 'LIT':
        I = (cat['DIAM_LIT'] > 0.) * (cat['DIAM_LIT_REF'] != 'RC3')
        if np.any(I):
            diam[I] = cat[I]['DIAM_LIT'] * 60. # [arcsec]
            ba[I] = cat[I]['BA_LIT']
            pa[I] = cat[I]['PA_LIT']
    elif ref == 'HYPERLEDA':
        I = cat['DIAM_HYPERLEDA'] > 0.
        if np.any(I):
            diam[I] = cat[I]['DIAM_HYPERLEDA'] * 60. # [arcsec]
            ba[I] = cat[I]['BA_HYPERLEDA']
            pa[I] = cat[I]['PA_HYPERLEDA']

    # clean up missing values of BA and PA
    ba[ba < 0.] = 1.
    pa[pa < 0.] = 0.
    
    return diam, ba, pa


def choose_geometry(cat, mindiam=152*0.262):
    """Choose an object's geometry, selecting between the
    NED-assembled (literature) values (DIAM, BA, PA), values from the
    SGA2020 (DIAM_SGA2020, BA_SGA2020, PA_SGA2020), and HyperLeda's
    values (DIAM_HYPERLEDA, BA_HYPERLEDA, PA_HYPERLEDA).

    mindiam is ~40 arcsec

    Default values of BA and PA are 1.0 and 0.0.

    """
    nobj = len(cat)
    diam = np.zeros(nobj) # [arcsec]
    ba = np.ones(nobj)
    pa = np.zeros(nobj)
    ref = np.zeros(nobj, '<U9')

    # choose RC3 literature
    I = np.logical_and.reduce((diam <= 0., cat['DIAM_LIT_REF'] == 'RC3', cat['DIAM_LIT'] > 0.))
    #I = np.logical_and.reduce((diam <= 0., cat['DIAM_LIT'] > 0., cat['DIAM_HYPERLEDA'] < 0.))
    if np.any(I):
        diam[I] = cat[I]['DIAM_LIT'] * 60. # [arcsec]
        ba[I] = cat[I]['BA_LIT']
        pa[I] = cat[I]['PA_LIT']
        ref[I] = 'RC3'

    # choose SGA2020
    I = np.logical_and(diam <= 0., cat['DIAM_SGA2020'] > 0.)
    if np.any(I):
        diam[I] = cat[I]['DIAM_SGA2020'] * 60. # [arcsec]
        ba[I] = cat[I]['BA_SGA2020']
        pa[I] = cat[I]['PA_SGA2020']
        ref[I] = 'SGA2020'

    # choose HyperLeda
    I = np.logical_and(diam <= 0., cat['DIAM_HYPERLEDA'] > 0.)
    #I = np.logical_and.reduce((diam <= 0., cat['DIAM_LIT'] < 0., cat['DIAM_HYPERLEDA'] > 0.))
    if np.any(I):
        diam[I] = cat[I]['DIAM_HYPERLEDA'] * 60. # [arcsec]
        ba[I] = cat[I]['BA_HYPERLEDA']
        pa[I] = cat[I]['PA_HYPERLEDA']
        ref[I] = 'HYPERLEDA'

    # choose literature
    I = np.logical_and(diam <= 0., cat['DIAM_LIT'] > 0.)
    #I = np.logical_and.reduce((diam <= 0., cat['DIAM_LIT'] > 0., cat['DIAM_HYPERLEDA'] < 0.))
    if np.any(I):
        diam[I] = cat[I]['DIAM_LIT'] * 60. # [arcsec]
        ba[I] = cat[I]['BA_LIT']
        pa[I] = cat[I]['PA_LIT']
        ref[I] = cat[I]['DIAM_LIT_REF']

    # set a minimum floor on the diameter
    I = diam < mindiam
    if np.any(I):
        diam[I] = mindiam

    # special-cases - north
    S = [
        'WISEA J151427.24+604725.4', # not in HyperLeda; basic-data diameter is 12.11 arcmin!
        ]
    I = np.isin(cat['OBJNAME'], S)
    if np.any(I):
        diam[I] = mindiam

    # clean up missing values of BA and PA
    ba[ba < 0.] = 1.
    pa[pa < 0.] = 0.

    return diam, ba, pa, ref

        
def get_basic_geometry(cat, galaxy_column='OBJNAME', verbose=False):
    """From a catalog containing magnitudes, diameters, position angles, and
    ellipticities, return a "basic" value for each property.

    Priority order: RC3, TWOMASS, SDSS, ESO, NED/BASIC

    """
    from astropy.table import Table
    nobj = len(cat)

    basic = Table()
    basic['GALAXY'] = cat[galaxy_column].value

    # default
    magcol = 'MAG_LIT'
    diamcol = 'DIAM_LIT'

    # HyperLeda
    if 'LOGD25' in cat.columns:
        ref = 'HYPERLEDA'
        magcol = f'MAG_{ref}'
        diamcol = f'DIAM_{ref}'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'BT'
                band = 'B'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'LOGD25'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = 0.1 * 10.**cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'LOGR25'
                I = ~np.isnan(cat[col]) * (cat[col] != 0.)
                if np.sum(I) > 0:
                    val[I] = 10.**(-cat[col][I])
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_{ref}'] = val
            basic[f'{prop.upper()}_{ref}_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_{ref}'] = val_band
            
    # SGA2020
    elif 'D26' in cat.columns:
        ref = 'SGA2020'
        magcol = f'MAG_{ref}'
        diamcol = f'DIAM_{ref}'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'R_MAG_SB26'
                band = 'R'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'D26'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'BA'
                I = ~np.isnan(cat[col]) * (cat[col] != 0.)
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_{ref}'] = val
            basic[f'{prop.upper()}_{ref}_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_{ref}'] = val_band
    # LVD
    elif 'RHALF' in cat.columns:
        ref = 'LVD'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'APPARENT_MAGNITUDE_V'
                band = 'V'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'RHALF' # [arcmin]
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I] * 2. * 2. # half-light-->full-light; radius-->diameter
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'ELLIPTICITY' # =1-b/a
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = 1. - cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'POSITION_ANGLE'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I] % 180 # put in the range [0, 180]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

    # custom
    elif 'DIAM' in cat.columns:
        ref = 'CUSTOM'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'MAG'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = cat[f'{col}_BAND'][I]
            elif prop == 'diam':
                col = 'DIAM' # [arcmin]
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'BA'
                I = cat[col] != -99.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = cat[col] != -99.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band
    # NED
    else:
        for prop in ('mag', 'diam', 'ba', 'pa'):
            if prop == 'mag':
                refs = ('SDSS', 'TWOMASS', 'RC3')
                bands = ('R', 'K', 'B')
            else:
                refs = ('ESO', 'SDSS', 'TWOMASS', 'RC3')
                bands = ('B', 'R', 'K', 'B')
            nref = len(refs)

            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            #allI = np.zeros((nobj, nref), bool)
            for iref, (ref, band) in enumerate(zip(refs, bands)):
                if prop == 'mag':
                    col = f'{ref}_{band}'
                else:
                    col = f'{ref}_{prop.upper()}_{band}'
                I = cat[col] > 0.
                #allI[:, iref] = I

                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

        # supplement any missing values with the "BASIC" data
        I = (basic['MAG_LIT'] <= 0.) * (cat['BASIC_MAG'] > 0.)
        if np.any(I):
            basic['MAG_LIT'][I] = cat['BASIC_MAG'][I]
            basic['BAND_LIT'][I] = 'V'

        I = (basic['DIAM_LIT'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.)
        if np.any(I):
            basic['DIAM_LIT'][I] = cat['BASIC_DMAJOR'][I]
            basic['DIAM_LIT_REF'][I] = 'BASIC'

        I = (basic['BA_LIT'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.) * (cat['BASIC_DMINOR'] > 0.)
        if np.any(I):
            basic['BA_LIT'][I] = cat['BASIC_DMINOR'][I] / cat['BASIC_DMAJOR'][I]
            basic['BA_LIT_REF'][I] = 'BASIC'

    # summarize
    if verbose:
        M = basic[magcol] > 0.
        D = basic[diamcol] > 0.
        print(f'Derived photometry for {np.sum(M):,d}/{nobj:,d} objects and ' + \
              f'diameters for {np.sum(D):,d}/{nobj:,d} objects.')

    return basic


def parent_datamodel(nobj):
    """Initialize the data model for the parent-nocuts sample.

    """
    parent = Table()
    parent['OBJNAME'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_NED'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_HYPERLEDA'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_NEDLVS'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_SGA2020'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_LVD'] = np.zeros(nobj, '<U30')
    parent['OBJTYPE'] = np.zeros(nobj, '<U6')
    parent['MORPH'] = np.zeros(nobj, '<U20')
    parent['BASIC_MORPH'] = np.zeros(nobj, '<U40')

    parent['RA'] = np.zeros(nobj, 'f8') -99.
    parent['DEC'] = np.zeros(nobj, 'f8') -99.
    parent['RA_NED'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_NED'] = np.zeros(nobj, 'f8') -99.
    parent['RA_HYPERLEDA'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_HYPERLEDA'] = np.zeros(nobj, 'f8') -99.
    parent['RA_NEDLVS'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_NEDLVS'] = np.zeros(nobj, 'f8') -99.
    parent['RA_SGA2020'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_SGA2020'] = np.zeros(nobj, 'f8') -99.
    parent['RA_LVD'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_LVD'] = np.zeros(nobj, 'f8') -99.

    parent['Z'] = np.zeros(nobj, 'f8') -99.
    parent['Z_NED'] = np.zeros(nobj, 'f8') -99.
    parent['Z_HYPERLEDA'] = np.zeros(nobj, 'f8') -99.
    parent['Z_NEDLVS'] = np.zeros(nobj, 'f8') -99.

    parent['PGC'] = np.zeros(nobj, '<i8') -99
    parent['ESSENTIAL_NOTE'] = np.zeros(nobj, '<U80')

    parent['MAG_LIT'] = np.zeros(nobj, 'f4') -99.
    parent['MAG_LIT_REF'] = np.zeros(nobj, '<U9')
    parent['BAND_LIT'] = np.zeros(nobj, '<U1')
    parent['DIAM_LIT'] = np.zeros(nobj, 'f4') -99.
    parent['DIAM_LIT_REF'] = np.zeros(nobj, '<U9')
    parent['BA_LIT'] = np.zeros(nobj, 'f4') -99.
    parent['BA_LIT_REF'] = np.zeros(nobj, '<U9')
    parent['PA_LIT'] = np.zeros(nobj, 'f4') -99.
    parent['PA_LIT_REF'] = np.zeros(nobj, '<U9')

    parent['MAG_HYPERLEDA'] = np.zeros(nobj, 'f4') -99.
    parent['BAND_HYPERLEDA'] = np.zeros(nobj, '<U1')
    parent['DIAM_HYPERLEDA'] = np.zeros(nobj, 'f4') -99.
    parent['BA_HYPERLEDA'] = np.zeros(nobj, 'f4') -99.
    parent['PA_HYPERLEDA'] = np.zeros(nobj, 'f4') -99.

    parent['MAG_SGA2020'] = np.zeros(nobj, 'f4') -99.
    parent['BAND_SGA2020'] = np.zeros(nobj, '<U1')
    parent['DIAM_SGA2020'] = np.zeros(nobj, 'f4') -99.
    parent['BA_SGA2020'] = np.zeros(nobj, 'f4') -99.
    parent['PA_SGA2020'] = np.zeros(nobj, 'f4') -99.

    parent['ROW_HYPERLEDA'] = np.zeros(nobj, '<i8') -99
    parent['ROW_NEDLVS'] = np.zeros(nobj, '<i8') -99
    parent['ROW_SGA2020'] = np.zeros(nobj, '<i8') -99
    parent['ROW_LVD'] = np.zeros(nobj, '<i8') -99
    parent['ROW_CUSTOM'] = np.zeros(nobj, '<i8') -99

    return parent


def choose_primary(group, verbose=False, keep_all_mergers=False):
    """Choose the primary member of a group.

    keep_all is helpful for returning a group catalog without dropping any
    sources.

    if keep_all_mergers=True then always keep {GPair,GTrpl} sources, even
      if they do not have a diameter.

    if allow_vetos=True then do not drop systems that are in a 'veto' list,
      even if they overlap with another source.

    """
    if keep_all_mergers:
        IM = np.logical_or(group['OBJTYPE'] == 'GPair', group['OBJTYPE'] == 'GTrpl')
        IG = group['OBJTYPE'] == 'G'
    else:
        IG = np.logical_or(group['OBJTYPE'] == 'G', group['OBJTYPE'] == 'GPair', group['OBJTYPE'] == 'GTrpl')

    #IG = np.logical_or.reduce((group['OBJTYPE'] == 'G', group['OBJTYPE'] == 'GPair', group['OBJTYPE'] == 'GTrpl'))
    ID = np.vstack((group['DIAM_LIT'] != -99., group['DIAM_HYPERLEDA'] != -99.)).T
    IZ = group['Z'] != -99.
    IS = group['SEP'] == 0.

    mask1 = IG * np.any(ID, axis=1)      # objtype=G and any diameter
    mask2 = IG * np.all(ID, axis=1)      # objtype=G and both diameters
    mask3 = IG * np.all(ID, axis=1) * IZ # objtype=G, both diameters, and a redshift
    mask4 = np.all(ID, axis=1) * IZ      # both diameters and a redshift
    mask5 = np.all(ID, axis=1) * IS      # both diameters and separation=0 (usually PGC is a minimum)
    mask6 = np.all(ID, axis=1)           # both diameters
    mask7 = np.any(ID, axis=1) * IS      # either diameter and separation=0
    mask8 = IS                           # separation=0
    mask9 = IG                           # objtype=G

    if keep_all_mergers:
        mask0 = IM # objtype={GPair,GTrpl}
        allmasks = (mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8)
    else:
        allmasks = (mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9)

    for mask in allmasks:
        keep = np.where(mask)[0]
        drop = np.where(~mask)[0]
        if len(keep) == 1:
            keep, drop = np.where(mask)[0], np.where(~mask)[0]
            return keep, drop

    print('Warning: cases 1-9 failed; choosing by prefix.')
    prefer_prefix = ['NGC', 'UGC', 'IC', 'MCG', 'CGCG', 'ESO',
                     'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
    prefix = np.array(list(zip(*np.char.split(group['OBJNAME'].value, ' ').tolist()))[0])
    mask = np.array([pre in prefer_prefix for pre in prefix])
    keep = np.where(mask)[0]
    drop = np.where(~mask)[0]
    if len(keep) == 1:
        print(group['OBJNAME', 'OBJTYPE', 'RA', 'DEC', 'DIAM_LIT', 'DIAM_HYPERLEDA', 'Z', 'PGC', 'SEP'])
        keep, drop = np.where(mask)[0], np.where(~mask)[0]
        return keep, drop

    print('Warning: choosing by prefix failed; choosing by minimum separation.')
    print(group['OBJNAME', 'OBJTYPE', 'RA', 'DEC', 'DIAM_LIT', 'DIAM_HYPERLEDA', 'Z', 'PGC', 'SEP'])
    print()
    keep = np.atleast_1d(group['SEP'].argmin())
    return keep, np.delete(np.arange(len(group)), keep)
    #indx = np.arange(len(group))
    #return indx[:1], indx[1:]


def resolve_close(cat, refcat, maxsep=1., keep_all=False, allow_vetos=False,
                  keep_all_mergers=False, objname_column='OBJNAME',
                  trim=True, verbose=False):
    """Resolve close objects.

    maxsep in arcsec
    cat - smaller catalog
    refcat - full catalog

    """
    VETO = [
        '2MASX J15134005+2607307',  # overlaps with 3C 315, but the coordinates for 3C 315 are wrong
    ]

    allmatches = match_radec(cat['RA'].value, cat['DEC'].value,
                             refcat['RA'].value, refcat['DEC'].value,
                             maxsep/3600., indexlist=True, notself=False)

    nobj = 0
    allindx_cat, allindx_refcat = [], []
    for iobj, onematch in enumerate(allmatches):
        if onematch is None:
            continue
        nmatch = len(onematch)
        if nmatch > 1:
            nobj += nmatch
            allindx_cat.append(iobj)
            allindx_refcat.append(onematch)

    if verbose:
        maxname = len(max(refcat[objname_column], key=len))
        maxtyp = len(max(refcat['OBJTYPE'], key=len))

    refcat['GROUP_ID'] = np.zeros(len(refcat), np.int32) - 99
    refcat['PRIMARY'] = np.ones(len(refcat), bool)
    refcat['NGROUP'] = np.ones(len(refcat), np.int16)
    refcat['SEPARATION'] = np.zeros(len(refcat), 'f4')
    refcat['DONE'] = np.zeros(len(refcat), bool)

    for igroup, (indx_cat, indx_refcat) in enumerate(zip(allindx_cat, allindx_refcat)):
        if verbose and (igroup % 500 == 0):
            print(f'Working on group {igroup+1:,d}/{len(allindx_refcat):,d}')
        indx_cat = np.array(indx_cat)
        indx_refcat = np.array(indx_refcat)
        if np.all(refcat['DONE'][indx_refcat]):
            continue

        group = refcat[indx_refcat]
        dtheta = arcsec_between(Table(cat[indx_cat])['RA'], Table(cat[indx_cat])['DEC'],
                                group['RA'].value, group['DEC'].value)
        group['SEP'] = dtheta.astype('f4')
        ngroup = len(group)

        refcat['GROUP_ID'][indx_refcat] = igroup
        refcat['NGROUP'][indx_refcat] = ngroup
        refcat['SEPARATION'][indx_refcat] = dtheta
        refcat['DONE'][indx_refcat] = True

        if keep_all:
            primary = np.arange(ngroup)
            drop = np.array([])
        else:
            primary, drop = choose_primary(group, verbose=verbose, keep_all_mergers=keep_all_mergers)
            refcat['PRIMARY'][indx_refcat[drop]] = False

        #if verbose and (np.any(group['OBJTYPE'] == 'GPair') or np.any(group['OBJTYPE'] == 'GTrpl')):
        if verbose:
            for ii in primary:
                print('Keep: '+'{name: <{W}}'.format(name=group[ii][objname_column], W=maxname)+': ' + \
                      '{typ: <{W}}'.format(typ=group[ii]["OBJTYPE"], W=maxtyp)+', ' + \
                      f'PGC={group[ii]["PGC"]}, z={group[ii]["Z"]:.5f}, D={group[ii]["DIAM_LIT"]:.2f}, ' + \
                      f'D(LEDA)={group[ii]["DIAM_HYPERLEDA"]:.2f} arcmin, (ra,dec)={group[ii]["RA"]:.6f},' + \
                      f'{group[ii]["DEC"]:.6f}')
            for ii in drop:
                print('Drop: '+'{name: <{W}}'.format(name=group[ii][objname_column], W=maxname)+': ' + \
                      '{typ: <{W}}'.format(typ=group[ii]["OBJTYPE"], W=maxtyp)+', ' + \
                      f'PGC={group[ii]["PGC"]}, z={group[ii]["Z"]:.5f}, D={group[ii]["DIAM_LIT"]:.2f}, ' + \
                      f'D(LEDA)={group[ii]["DIAM_HYPERLEDA"]:.2f} arcmin, sep={group[ii]["SEP"]:.3f} arcsec')

        # check for vetos
        if allow_vetos:
            for veto in VETO:
                I = np.where(np.isin(refcat[indx_refcat][objname_column], veto))[0]
                if len(I) == 1:
                    if refcat[indx_refcat[I]]['PRIMARY'] == False:
                        print(f'Restoring vetoed object {refcat[indx_refcat[I[0]]][objname_column]}')
                        refcat['PRIMARY'][indx_refcat[I]] = True
        if verbose:
            print()

    ugroup = np.unique(refcat["GROUP_ID"])
    ugroup = ugroup[ugroup != -99]

    print(f'Found {len(ugroup):,d} group(s) with ' + \
          f'({np.sum(refcat["GROUP_ID"] != -99):,d}/{len(refcat):,d} ' + \
          f'unique objects) within {maxsep:.1f} arcsec.')

    #check = refcat[refcat['GROUP_ID'] != -99]
    #check = check[np.argsort(check['GROUP_ID'])]
    #drop = refcat[(refcat['GROUP_ID'] != -99) * (refcat['PRIMARY'] == False)]
    #prefix = np.array([objname.split(' ')[0] for objname in drop['OBJNAME']])

    if trim:
        out = refcat[refcat['PRIMARY']]
        out.remove_columns(['GROUP_ID', 'PRIMARY', 'NGROUP', 'SEPARATION', 'DONE'])
        return out
    else:
        refcat.remove_column('DONE')
        return refcat


def match_to(A, B, check_for_dups=True):
    """Return indexes where B matches A, holding A in place.

    Parameters
    ----------
    A : :class:`~numpy.ndarray` or `list`
        Array of integers to match TO.
    B : :class:`~numpy.ndarray` or `list`
        Array of integers matched to `A`
    check_for_dups : :class:`bool`, optional, defaults to ``True``
        If ``True`` trigger an exception if there are duplicates in
        either of the `A` or `B` arrays. Passing ``False`` for
        `check_for_dups` isn't recommended, but is retained to facilitate
        comparisons against earlier versions of the function.

    Returns
    -------
    :class:`~numpy.ndarray`
        The integer indexes in A that B matches to.

    Notes
    -----
    - Result should be such that for ii = match_to(A, B)
      np.all(A[ii] == B) is True.
    - We're looking up locations of B in A so B can be
      a shorter array than A (provided the B->A matches are
      unique) but A cannot be a shorter array than B.
    """
    # ADM grab the integer matchers.
    Aii, Bii = match(A, B, check_for_dups=check_for_dups)

    # ADM return, restoring the original order.
    return Aii[np.argsort(Bii)]


def match(A, B, check_for_dups=True):
    """Return matching indexes for two arrays on a unique key.

    Parameters
    ----------
    A : :class:`~numpy.ndarray` or `list`
        An array of integers.
    B : :class:`~numpy.ndarray` or `list`
        An array of integers.
    check_for_dups : :class:`bool`, optional, defaults to ``True``
        If ``True`` trigger an exception if there are duplicates in
        either of the `A` or `B` arrays. Passing ``False`` for
        `check_for_dups` isn't recommended, but is retained to facilitate
        comparisons against earlier versions of the function.

    Returns
    -------
    :class:`~numpy.ndarray`
        The integer indexes in A that match to B.
    :class:`~numpy.ndarray`
        The integer indexes in B that match to A.

    Notes
    -----
    - Result should be such that for Aii, Bii = match(A, B)
      np.all(A[Aii] == B[Bii]) is True.
    - Only works if there is a unique mapping from A->B, i.e
      if A and B do NOT contain duplicates. This is explicitly checked if
      `check_for_dups` is ``True``
    - h/t Anand Raichoor `by way of Stack Overflow`_.
    """
    # AR sorting A,B
    tmpA = np.sort(A)
    tmpB = np.sort(B)

    # ADM via AR rapid check for duplicates in either array.
    if check_for_dups:
        n_Adups = np.count_nonzero(tmpA[1:] == tmpA[:-1])
        n_Bdups = np.count_nonzero(tmpB[1:] == tmpB[:-1])
        msg = []
        if n_Adups > 0:
            msg.append("Array A has {} duplicates".format(n_Adups))
        if n_Bdups > 0:
            msg.append("Array B has {} duplicates".format(n_Bdups))
        if len(msg) > 0:
            msg = "; ".join(msg)
            print(msg)
            raise ValueError(msg)

    # AR mask equivalent to np.in1d(A, B) for unique elements.
    maskA = (
        np.searchsorted(tmpB, tmpA, "right") - np.searchsorted(tmpB, tmpA, "left")
    ) == 1
    maskB = (
        np.searchsorted(tmpA, tmpB, "right") - np.searchsorted(tmpA, tmpB, "left")
    ) == 1

    # AR to get back to original indexes
    return np.argsort(A)[maskA], np.argsort(B)[maskB]


def weighted_partition(weights, n):
    '''
    Partition `weights` into `n` groups with approximately same sum(weights)

    Args:
        weights: array-like weights
        n: number of groups

    Returns list of lists of indices of weights for each group

    Notes:
        compared to `dist_discrete_all`, this function allows non-contiguous
        items to be grouped together which allows better balancing.

    '''
    #- sumweights will track the sum of the weights that have been assigned
    #- to each group so far
    sumweights = np.zeros(n, dtype=float)

    #- Initialize list of lists of indices for each group
    groups = list()
    for i in range(n):
        groups.append(list())

    #- Assign items from highest weight to lowest weight, always assigning
    #- to whichever group currently has the fewest weights
    weights = np.asarray(weights)
    for i in np.argsort(-weights):
        j = np.argmin(sumweights)
        groups[j].append(i)
        sumweights[j] += weights[i]

    assert len(groups) == n

    return groups


def radec_to_name(target_ra, target_dec, prefix='SGA2025'):
    """Convert the right ascension and declination of an object into a
    disk-friendly "name", for reference in publications.  Length of
    `target_ra` and `target_dec` must be the same if providing an
    array or list.

    Parameters
    ----------
    target_ra: array of :class:`~numpy.float64`
        Right ascension in degrees of target object(s). Can be float, double,
        or array/list of floats or doubles.
    target_dec: array of :class:`~numpy.float64`
        Declination in degrees of target object(s). Can be float, double,
        or array/list of floats or doubles.

    Returns
    -------
    array of :class:`str`
        Names referring to the input target RA and DEC's. Array is the
        same length as the input arrays.

    Raises
    ------
    ValueError
        If any input values are out of bounds.

    Notes
    -----
    Written by A. Kremin (LBNL) for DESI. Taken entirely from
    desiutil.names.radec_to_desiname.

    """
    # Convert to numpy array in case inputs are scalars or lists
    target_ra, target_dec = np.atleast_1d(target_ra), np.atleast_1d(target_dec)

    base_tests = [('NaN values', np.isnan),
                  ('Infinite values', np.isinf),]
    inputs = {'target_ra': {'data': target_ra,
                            'tests': base_tests + [('RA not in range [0, 360)', lambda x: (x < 0) | (x >= 360))]},
              'target_dec': {'data': target_dec,
                             'tests': base_tests + [('Dec not in range [-90, 90]', lambda x: (x < -90) | (x > 90))]}}
    for coord in inputs:
        for message, check in inputs[coord]['tests']:
            if check(inputs[coord]['data']).any():
                raise ValueError(f"{message} detected in {coord}!")

    # Number of decimal places in final naming convention
    precision = 4

    # Truncate decimals to the given precision
    ratrunc = np.trunc((10.**precision) * target_ra).astype(int).astype(str)
    dectrunc = np.trunc((10.**precision) * target_dec).astype(int).astype(str)

    # Loop over input values and create the name as DESINAME as: DESI JXXX.XXXX+/-YY.YYYY
    # Here J refers to J2000, which isn't strictly correct but is the closest
    #   IAU compliant term
    names = []
    for ra, dec in zip(ratrunc, dectrunc):
        zra = ra.zfill(7)
        name = f'{prefix} J' + zra[:-precision] + '.' + zra[-precision:]
        # Positive numbers need an explicit "+" while negative numbers
        #   already have a "-".
        # zfill works properly with '-' but counts it in number of characters
        #   so need one more
        if dec.startswith('-'):
            zdec = dec.zfill(7)
            name += zdec[:-precision] + '.' + zdec[-precision:]
        else:
            zdec = dec.zfill(6)
            name += '+' + zdec[:-precision] + '.' + zdec[-precision:]
        names.append(name)

    return np.array(names)
