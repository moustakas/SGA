"""
SGA.ellipse
===========

Code to perform ellipse photometry.

"""
import numpy as np

#print('Try using cntrd in lieu of find_galaxy')
#print('https://github.com/djones1040/PythonPhot/blob/master/PythonPhot/cntrd.py')


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

    # [1] LVD
    I = np.logical_and.reduce((diam <= 0., cat['DIAM_LIT_REF'] == 'LVD', cat['DIAM_LIT'] > 0.))
    if np.any(I):
        diam[I] = cat[I]['DIAM_LIT'] * 60. # [arcsec]
        ba[I] = cat[I]['BA_LIT']
        pa[I] = cat[I]['PA_LIT']
        ref[I] = 'LVD'

    # [2] RC3
    I = np.logical_and.reduce((diam <= 0., cat['DIAM_LIT_REF'] == 'RC3', cat['DIAM_LIT'] > 0.))
    if np.any(I):
        diam[I] = cat[I]['DIAM_LIT'] * 60. # [arcsec]
        ba[I] = cat[I]['BA_LIT']
        pa[I] = cat[I]['PA_LIT']
        ref[I] = 'RC3'

    # [3] SGA2020
    I = np.logical_and(diam <= 0., cat['DIAM_SGA2020'] > 0.)
    if np.any(I):
        diam[I] = cat[I]['DIAM_SGA2020'] * 60. # [arcsec]
        ba[I] = cat[I]['BA_SGA2020']
        pa[I] = cat[I]['PA_SGA2020']
        ref[I] = 'SGA2020'

    # [4] HyperLeda
    I = np.logical_and(diam <= 0., cat['DIAM_HYPERLEDA'] > 0.)
    if np.any(I):
        diam[I] = cat[I]['DIAM_HYPERLEDA'] * 60. # [arcsec]
        ba[I] = cat[I]['BA_HYPERLEDA']
        pa[I] = cat[I]['PA_HYPERLEDA']
        ref[I] = 'HYPERLEDA'

    # [5] literature
    I = np.logical_and(diam <= 0., cat['DIAM_LIT'] > 0.)
    #I = np.logical_and.reduce((diam <= 0., cat['DIAM_LIT'] > 0., cat['DIAM_HYPERLEDA'] < 0.))
    if np.any(I):
        diam[I] = cat[I]['DIAM_LIT'] * 60. # [arcsec]
        ba[I] = cat[I]['BA_LIT']
        pa[I] = cat[I]['PA_LIT']
        ref[I] = cat[I]['DIAM_LIT_REF']

    # set a minimum floor on the diameter
    I = diam <= mindiam
    if np.any(I):
        diam[I] = mindiam

    ## special-cases - north
    #S = [
    #    'WISEA J151427.24+604725.4', # not in HyperLeda; basic-data diameter is 12.11 arcmin!
    #    ]
    #I = np.isin(cat['OBJNAME'], S)
    #if np.any(I):
    #    diam[I] = mindiam

    # clean up missing values of BA and PA
    ba[ba < 0.] = 1.
    pa[pa < 0.] = 0.

    return diam, ba, pa, ref

        
