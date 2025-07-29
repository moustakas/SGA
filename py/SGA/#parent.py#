"""
SGA.parent
==========

Code for defining the SGA parent sample.

"""
import os, pdb
import numpy as np
import fitsio
from importlib import resources
from collections import Counter
from astropy.table import Table, vstack
from astrometry.libkd.spherematch import match_radec

from SGA.io import sga_dir
from SGA.geometry import get_basic_geometry
from SGA.sky import match, choose_primary, resolve_close
from SGA.qa import qa_skypatch, multipage_skypatch

from SGA.logger import log


def drop_by_prefix(drop_prefix, allprefixes, pgc=None, diam=None, objname=None,
                   VETO=None, reverse=False, verbose=False):
    """Drop sources according to their name prefix. Most/all of these have
    been visually inspected. However, don't drop a source if it has PGC
    number and at least one diameter.

    """
    I = drop_prefix == allprefixes
    if pgc is not None and diam is not None:
        G = (pgc != -99) * (diam != -99.)
        I *= ~G
    elif diam is not None:
        G = (diam != -99.)
        I *= ~G

    if objname is not None and VETO is not None:
        I *= ~np.isin(objname, VETO)

    if reverse:
        I = np.where(~I)[0]
    else:
        I = np.where(I)[0]

    if verbose:
        if reverse:
            log.info(f'Keeping {len(I):,d} object(s) without prefix {drop_prefix}.')
        else:
            log.info(f'Dropping {len(I):,d} object(s) with prefix {drop_prefix}.')
    return I


def remove_by_prefix(fullcat, merger_type=None, merger_has_diameter=False, build_qa=False,
                     cleanup=True, verbose=False, overwrite=False):
    """Remove merger_types based on their name "prefix" (or reference), most of
    which are ultra-faint or galaxy groups (i.e., not galaxies).

    """
    if merger_type is not None:
        suffix = merger_type
        qasuffix = merger_type
        #allprefix = np.array(list(zip(*np.char.split(fullcat['OBJNAME'].value, ' ').tolist()))[0])
        T = fullcat[fullcat['OBJTYPE'] == merger_type]
    else:
        suffix = ''
        qasuffix = 'uncommon-prefix'
        T = fullcat.copy()

    log.info(f'Analyzing the prefix frequency of {len(T):,d}/{len(fullcat):,d} objects.')

    diam = np.max((T['DIAM_LIT'].value, T['DIAM_HYPERLEDA'].value), axis=0)
    if merger_type is not None:
        if merger_has_diameter:
            I = np.where(diam != -99)[0]
            log.info(f'Trimming to {len(I):,d}/{len(T):,d} {merger_type} with an estimated diameter.')
            diam = None
        else:
            I = np.where(diam == -99)[0]
            diam = diam[I]
            log.info(f'Trimming to {len(I):,d}/{len(T):,d} {merger_type} with no diameter.')
        T = T[I]

    objname = T['OBJNAME'].value
    prefix = np.array(list(zip(*np.char.split(T['OBJNAME'].value, ' ').tolist()))[0])

    # do not cut on PGC number
    #pgc = cat['PGC'].value
    pgc = None

    C = Counter(prefix)
    #log.info(C.most_common())
    #multipage_skypatch(T[prefix == 'FLX'][:20], cat=T, clip=True, pngdir='/pscratch/sd/i/ioannis/tmp', jpgdir='/pscratch/sd/i/ioannis/tmp')

    drop_ignore_diam_prefixes = []
    match merger_type:
        case 'GTrpl':
            VETO = None
            if merger_has_diameter:
                drop_prefixes = []
                drop_ignore_diam_prefixes = []
            else:
                drop_prefixes = ['[BWH2007]', '2dFGRS', 'APMUKS(BJ)',
                                 'RSCG', 'MCG', 'LGG', 'LCLG', 'IRAS', 'PPS2', 'GALEXMSC',
                                 '[ALB2009]', 'WBL', 'KTS', 'ARP',
                                 'AM', 'V1CG', 'KTG', 'UGC', 'VII', 'VV', 'WISEA',
                                 'CGCG', 'LDCE', 'HDCE', 'USGC', 'UZC-CG', 'ESO', 'MLCG',
                                 'PM2GC', 'V1CG', '[SPS2007]', 'UZC-CG', 'FLASH', 
                                 'USGC', 'CB-20.07763', 'SSRS', 'APMBGC', 'WAS', 'VCC',
                                 '2MASS', 'CGPG', 'II', 'VI', 'I', 'IC', 'IV', 'SDSS',
                                 'GALEXASC', 'VIII', 'V', 'NGC', ]
            drop_references = []
        case 'GPair':
            VETO = ['ESO 347-IG 014', ]
            if merger_has_diameter:
                drop_prefixes = []
                drop_ignore_diam_prefixes = ['CGMW'] # all in the MW disk
            else:
                drop_prefixes = ['[PCM2000]', '[ATS2004]', '[BFH91]', '[PPC2002]', 
                                 '[vvd91a]', '2MASS', '2MASX', '2MASXi', '2MFGC', 'APMBGC',
                                 'CGPG', 'Cocoon', 'CSL', 'CTS', 'FCCB', 'FLASH', 'MESSIER', 'GIN',
                                 'VPCX', 'WAS', 'TOLOLO', 'SGC', 'KOS', 'PKS', 'PGC1', 'SARS',
                                 'MAPS-NGP', 'FOCA', 'LSBG', 'LSBC', 'PGC', 'MRK', 'UM',
                                 'UGCA', 'GALEX', 'GALEXMSC', 'VCC', 'USGC', 'MGC', 'LCRS',
                                 'I', 'II', 'III', 'IV', 'VI', 'IC', 'MCG', 'NGC', 'APMUKS(BJ)',
                                 'SDSS', 'V', 'KUG', 'GALEXASC', 'ARP', 'VV', 'IRAS', 'UGC',
                                 'VIII', '2dFGRS', 'VII', 'AM', 'CGMW', 'WISEA', 'KPG', 'CGCG',
                                 'HS', 'ESO']
            drop_references = []
        case _:
            VETO = ['HIPASS J1348-37', 'HIPASS J1131-31', 'HIPASS J1247-77', 'HIPASS J1337-39', # LVD systems
                    'HIPASS J1133-32', 'HIPASS J1351-47', 'AGC 198606', # LVD systems
                    'RCS 04020600360', 'NSA 142339', 'HIDEEP J1337-3320', # LVD systems
                    'ZwCl 0054.6-0127 18', 'ZwCl 0054.6-0127 16 NED02', 'ZwCl 0054.6-0127 16 NED01', # in UGC 00579 group
                    'ZwCl 0054.6-0127 31', # in UGC 00588 group
                    ]
            drop_prefixes = ['WINGS', 'SRGb', '[MD2000]', 'HeCS', 'ESP',
                             'SDSSCGB', 'NVSS', 'AGC', ]
            drop_ignore_diam_prefixes = [
                'Mr20:[BFW2006]', 'Mr19:[BFW2006]', 'Mr18:[BFW2006]',
                '[BFW2006]', '[IBG2003]', '[BJG2014]', 'OPH', 'COMAi', 'zCOSMOS',
                'CDWFS', 'NuSTAR', '4U', '2CXO', '1RXS', 'SSTSL2', '[MHH96]',
                'NRGs', 'NSA', '[KSP2018]', '[AHH96]', 'NEP', '[MKB2002]',
                '[PSP2017]', 'ZwCl', '[SAB2007]', 'COSMOS', '1256+27W01',
                'Subaru-UDG', 'S-CANDELS', 'GOODS', 'TKRS', 'DEEP2', 'VVDS',
                'GMASS', 'COMBO-17', 'CNOC2', 'RCS', 'CANDELS', 'ACS-GC',
                'UKIDSS:[WQF2009]', 'UDF:[CBS2006]', 'FDF', 'EDCSN', 'HDF:[WBD96]',
                'HDF:[LYF96]', 'HDF:[T2003]', 'HDFS:[RVB2005]', 'HDF:[CCH2004]', '[BRD2006]',
                'CFRS', 'UDF:[XPM2007]', 'COSMOS2015',
                'FIREWORKS', 'UDF:[BSK2006]', 'ADBS', 'AKARI', 'Bolocam', '[CIR2010]',
                'LH610MHz', 'RG', 'CB-19.04116', 'ISO_A2218_54', 'ISO_A2218_70',
                'HIDEEP', '[MGP2017]', 'Shadowy', 'VLANEP', 'TAFFY2', 'SMM',
                'Lock', 'SSSG', 'LQAC', 'HSG', 'HIZSS', 'HOLM', '[KLI2009]',
                'NSCS', 'H-KPAIR', 'KPAIR', 'HIPASS', 'ALFAZOA', 'HDF', '[MFB2005]',
                'RR', 'RXC', 'MODS', 'BAS', 'HIZOA', 'LaCoSSPAr', 'HIJASS',
                'AGESVC1', 'LRG', '[MLO2002]', '[H87]', '[ARN85]', '[LYY2017]',
                'SDWFS', '[DMS2015]', 'FLX', 'CDFN:[ABB2003]', 'PDCS', '[MOH2004]',
                'MS', '[SNS2015]', '[KEC2012]', '[LPG2013]', '6C', 'SDSS-II',
                'GNS', 'CFHIZELS-SM14', 'CASBaH', '[DYC2005]', 'WIG', 'SHELS',
                '[TGK2001]', '[CBR2014]', 'ANTL', '[TH2002]', '[CBR2013]', ]
            drop_references = [
                'MESSIER 087:[SRB2011]', 'ABELL 1656:[BDG95]', 'ABELL 1656:[EF2011]',
                # candidate globular clusters:
                'NGC 3607:[H2009]', 'NGC 5846:[H2009]', 'NGC 4494:[H2009]',
                'NGC 5845:[H2009]', 'NGC 4125:[H2009]',
                'NGC 4649:[KMZ2007]', 'NGC 3115:[KMZ2007]',
                ###
                'Cl J1604+4304:[PLO98]', 'Cl J1604+4303:[LPO98a]', 'Cl J1604+4321:[PLO2001]',
                'HERC-1:[MKK97]', 'CL 1601+4253:[DG92]',
                'W4+0+0:[DD2013]', 'W4+0+1:[DD2013]', 'W4+0-1:[DD2013]', 'W4+0-2:[DD2013]',
                'W4+1+0:[DD2013]', 'W4+1+1:[DD2013]', 'W4+1-1:[DD2013]', 'W4+1-2:[DD2013]',
                'W4+2+0:[DD2013]', 'W4+2-1:[DD2013]', 'W4+2-2:[DD2013]', 'W4-1+0:[DD2013]',
                'W4-1+1:[DD2013]', 'W4-1+2:[DD2013]', 'W4-1+3:[DD2013]', 'W4-1-1:[DD2013]',
                'W4-1-2:[DD2013]', 'W4-2+0:[DD2013]', 'W4-2+1:[DD2013]', 'W4-2+2:[DD2013]',
                'W4-2+3:[DD2013]', 'W4-3+0:[DD2013]', 'W4-3+1:[DD2013]', 'W4-3+2:[DD2013]',
                'W4-3+3:[DD2013]',
                "Stephan's Quintet:[XLC2003]", 'HCG 092:[DUM2012]', 'HCG 092:[KAG2014]',
                'HCG 090:[ZM98]', 'HCG 090:[DMG2011]',
                'ABELL 0671:[MK91]', 'ABELL 1656:[GMP83]', 'ABELL 3558:[KMD98]',
                'ABELL 3558:[MGP94]', 'ABELL 2218:[BOW83]', 'ABELL 2218:[LPS92]',
                'ABELL 2218:AMI2012', 'GEMS', 'SCOSMOS', 'PKS 0537-441:[HJN2003]',
                'PKS 0405-12:[PWC2006]', 'PKS 2005-489:[PWC2011]', 'PKS 1614+051:[HBS2015]',
                'PKS 2155-304:[FFD2016]', 'PKS 1934-63:[RHL2016]', 'PKS 2135-14:[HBC97]',
                'HS 1700+6416:[SSE2005]', 'NGC 4676:[BFB2004]', 'ABELL 2553:[QR95]',
                '[GC2020]', '[GDW2019]', 'ABELL 1689:[GH91]', 'ABELL 1689:[MBC96]',
                'ABELL 1689:[TCG90]', 'ABELL 1677:[AEF92]', 'ABELL 3295:[VCC89]',
                'ABELL 1205:[QR95]', 'ABELL 1413:[CSP2012]', 'ABELL 1413:[TFM92]',
                'ABELL 1413:[MPC97]', ]
            # {'ABELL 0116:KYDISC',
            # 'ABELL 0646:KYDISC',
            # 'ABELL 0655:KYDISC',
            # 'ABELL 0667:KYDISC',
            # 'ABELL 0690:KYDISC',
            # 'ABELL 1126:KYDISC',
            # 'ABELL 1139:KYDISC',
            # 'ABELL 1146:KYDISC',
            # 'ABELL 1278:KYDISC',
            # 'ABELL 2061:KYDISC',
            # 'ABELL 2249:KYDISC',
            # 'ABELL 2589:KYDISC',
            # 'ABELL 3574:KYDISC',
            # 'ABELL 3659:KYDISC'}

    I = []
    # drop by prefix (but not if there's a measured diameter)
    if len(drop_prefixes) > 0:
        for drop_prefix in drop_prefixes:
            Idrop = drop_by_prefix(drop_prefix, prefix, pgc=pgc, diam=diam,
                                   objname=objname, VETO=VETO, verbose=False)
            log.info(f'Dropping {len(Idrop):,d} sources with prefix {drop_prefix} and no diameter.')
            I.append(Idrop)

    # drop by prefix, ignoring the measured diameter, if any
    if len(drop_ignore_diam_prefixes) > 0:
        for drop_prefix in drop_ignore_diam_prefixes:
            Idrop = drop_by_prefix(drop_prefix, prefix, pgc=None, diam=None,
                                   objname=objname, VETO=VETO, verbose=False)
            log.info(f'Dropping {len(Idrop):,d} sources with prefix {drop_prefix} ignoring diameter.')
            I.append(Idrop)

    # drop by reference
    if len(drop_references) > 0:
        for drop_reference in drop_references:
            Idrop = np.flatnonzero(np.char.find(T['OBJNAME'].value, drop_reference) != -1)
            log.info(f'Dropping {len(Idrop):,d} {suffix} sources with reference {drop_reference}')
            I.append(Idrop)

    if len(I) > 0:
        I = np.unique(np.hstack(I))
        if verbose:
            log.info(f'Removing {len(I):,d}/{len(T):,d} {suffix} sources.')
        out = fullcat[~np.isin(fullcat['ROW_PARENT'], T[I]['ROW_PARENT'])]
    else:
        out = fullcat


    # optionally build QA
    if build_qa:
        width_arcsec = 120. # 30.
        ncol, nrow = 1, 1

        jpgdir = os.path.join(sga_dir(), 'parent', 'vi2', f'{qasuffix}-viewer')
        pngdir = os.path.join(sga_dir(), 'parent', 'vi2', f'{qasuffix}-png')

        qa_prefixes = [pre for pre in list(C.keys()) if pre not in drop_prefixes]
        #qa_prefixes = [pre for pre in list(C.keys()) if C[pre] <= 10 and pre not in drop_prefixes]

        #qa_prefixes = ['UGC', ]
        #obj='IC 1942' ; cc = out ; I=cc['OBJNAME'] == obj ; m1, m2, _ = match_radec(cc['RA'], cc['DEC'], cc[I]['RA'], cc[I]['DEC'], 75./3600.) ; cc[m1][cols]
        #prim=1 ; qa_skypatch(cc[m1[prim]], group=cc[m1])
        #m1, m2, _ = match_radec(fullcat['RA'], fullcat['DEC'], T[prefix == 'MGC']['RA'], T[prefix == 'MGC']['DEC'], 75./3600.)
        #m1, m2, _ = match_radec(fullcat['RA'], fullcat['DEC'], T[prefix == '[BFH91]']['RA'], T[prefix == '[BFH91]']['DEC'], 75./3600.)
        #fullcat[np.flatnonzero(np.char.find(fullcat['OBJNAME'].value, 'ABELL 2247') != -1)][cols]

        #for drop_prefix in ['III']:
        for drop_prefix in qa_prefixes:
        #for drop_prefix in drop_prefixes:

            Idrop = drop_by_prefix(drop_prefix, prefix, pgc=pgc, diam=diam,
                                   objname=objname, VETO=VETO, verbose=False)
            if len(Idrop) == 0:
                log.info(f'No {suffix} sources with prefix {drop_prefix}')
            else:
                log.info(f'Building QA for {len(Idrop):,d} {suffix} sources with prefix {drop_prefix}')
                pdffile = os.path.join(sga_dir(), 'parent', 'vi2', f'vi-{qasuffix}-{drop_prefix}.pdf')
                multipage_skypatch(T[Idrop], cat=out, width_arcsec=width_arcsec, ncol=ncol, nrow=nrow,
                                   jpgdir=jpgdir, pngdir=pngdir, pngsuffix=f'{qasuffix}-{drop_prefix}',
                                   pdffile=pdffile, clip=True, verbose=verbose, overwrite=overwrite,
                                   cleanup=cleanup)

    return out


def resolve_crossid_errors(fullcat, verbose=False, cleanup=False,
                           rebuild_file=False, build_qa=False):
    """Identify cross-identification errors.

    """
    from astrometry.libkd.spherematch import tree_build_radec, trees_match
    from astrometry.util.starutil_numpy import deg2dist, arcsec_between, degrees_between

    crossidfile = resources.files('SGA').joinpath('data/SGA2025/SGA2025-crossid-errors.csv')

    # Read or, optionally, rebuild the cross-id error file.
    if rebuild_file:
        VETO = [
            'NGC 3280', # will get dropped
            #'2MASS J04374625-2711389', # 2MASS J04374625-2711389 and WISEA J043745.83-271135.1 are distinct.
        ]

        # First, resolve 1-arcsec pairs **excluding** GPair and GTrpl systems.
        I = (fullcat['OBJTYPE'] != 'GPair') * (fullcat['OBJTYPE'] != 'GTrpl')
        cat = resolve_close(fullcat[I], fullcat[I], maxsep=1.5, allow_vetos=True, verbose=False)
        cat = vstack((cat, fullcat[~I]))
        cat = cat[np.argsort(cat['ROW_PARENT'])]
        #cat = resolve_close(fullcat, fullcat, maxsep=1.5, allow_vetos=True, verbose=False)

        #diam = np.max((cat['DIAM_LIT'].value, cat['DIAM_HYPERLEDA'].value), axis=0)

        # Next, algorithmically identify cross-identification errors: NED will
        # (sometimes) associate the properties from HyperLeda (PGC number,
        # diameter, etc.) with the *system* rather than with the appropriate
        # galaxy (which is another entry in NED).
        log.info(f'Writing {crossidfile}')
        F = open(crossidfile, 'w')
        F.write('objname_from,pgc_from,objname_to,pgc_to,dtheta_arcsec,comment\n')

        for system in ['GTrpl', 'GPair']:

            # Is there *another* source more than 1 arcsec away which is within
            # 1.5 arcsec of the *HyperLeda* coordinates? If so, add it to the
            # cross-id file.
            M = np.where((cat['OBJTYPE'] == system) * ~np.isin(cat['OBJNAME'], VETO))[0]

            # For speed, build a KD tree.
            kd_hyper = tree_build_radec(ra=cat[M]['RA_HYPERLEDA'], dec=cat[M]['DEC_HYPERLEDA'])
            kd = tree_build_radec(ra=cat['RA'], dec=cat['DEC'])
            I, J, _ = trees_match(kd_hyper, kd, deg2dist(1.5/3600.), notself=True, nearest=False)

            # I can contain duplicates, so build an M2 "list of lists" by looping
            M1, M2 = [], []
            for ii in np.unique(I):
                m1 = M[ii]
                m2 = np.sort(J[ii == I])
                # remove the primary
                m2 = m2[~np.isin(m2, m1)]
                M1.append(np.int64(m1))
                M2.append(np.int64(m2))

            uM2 = np.unique(np.hstack(M2))
            kd_hyper2 = tree_build_radec(ra=cat[uM2]['RA_HYPERLEDA'], dec=cat[uM2]['DEC_HYPERLEDA'])
            I2, J2, _ = trees_match(kd_hyper2, kd, deg2dist(1.5/3600.), notself=True, nearest=False)

            # tertiary match
            M3 = []
            for m1, m2 in zip(M1, M2):
                indx = np.where(np.isin(uM2[I2], m2))[0]
                if len(indx) == 0:
                    M3.append([])
                else:
                    m3 = J2[indx]
                    # remove the primary and secondary targets
                    m3 = m3[~np.isin(m3, np.hstack([m1, m2]))]
                    #cat[np.hstack((m1, m2, m3))]
                    M3.append(np.int64(m3))

            for m1, m2, m3 in zip(M1, M2, M3):
                #_, _m2, _ = match_radec(cat[m1]['RA_HYPERLEDA'], cat[m1]['DEC_HYPERLEDA'], cat['RA'], cat['DEC'], 1.5/3600.)
                #m2 = m2[~np.isin(m2, m1)] # remove self

                if len(m2) == 0:
                    continue
                elif len(m2) > 1:
                    #j1, j2, _ = match_radec(cat[m1]['RA_HYPERLEDA'], cat[m1]['DEC_HYPERLEDA'], cat['RA'], cat['DEC'], 75./3600.)
                    #multipage_skypatch(cat[m1], cat=cat[j2], ncol=1, nrow=1, overwrite=True)
                    group = cat[m2]
                    group['SEP'] = arcsec_between(cat[m1]['RA_HYPERLEDA'], cat[m1]['DEC_HYPERLEDA'], group['RA'], group['DEC'])
                    primary, drop = choose_primary(group)
                    if verbose:
                        objname_column = 'OBJNAME'
                        I = np.hstack((m1, m2))
                        maxname = len(max(cat[I][objname_column], key=len))
                        maxtyp = len(max(cat[I]['OBJTYPE'], key=len))
                        for ii in primary:
                            log.info('Keep: '+'{name: <{W}}'.format(name=group[ii][objname_column], W=maxname)+': ' + \
                                  '{typ: <{W}}'.format(typ=group[ii]["OBJTYPE"], W=maxtyp)+', ' + \
                                  f'PGC={group[ii]["PGC"]}, z={group[ii]["Z"]:.5f}, D={group[ii]["DIAM"]:.2f}, ' + \
                                  f'D(LEDA)={group[ii]["DIAM_HYPERLEDA"]:.2f} arcmin, (ra,dec)={group[ii]["RA"]:.6f},' + \
                                  f'{group[ii]["DEC"]:.6f}')
                        for ii in drop:
                            log.info('Drop: '+'{name: <{W}}'.format(name=group[ii][objname_column], W=maxname)+': ' + \
                                  '{typ: <{W}}'.format(typ=group[ii]["OBJTYPE"], W=maxtyp)+', ' + \
                                  f'PGC={group[ii]["PGC"]}, z={group[ii]["Z"]:.5f}, D={group[ii]["DIAM"]:.2f}, ' + \
                                  f'D(LEDA)={group[ii]["DIAM_HYPERLEDA"]:.2f} arcmin, sep={group[ii]["SEP"]:.3f} arcsec')

                    m2 = m2[primary]

                m2 = m2[0] # scalar

                # delta-theta between NED (old) and HyperLeda (adopted)
                dtheta_ned = arcsec_between(cat[m1]['RA'], cat[m1]['DEC'], cat[m1]['RA_HYPERLEDA'], cat[m1]['DEC_HYPERLEDA'])

                # objname_ned_from,objname_ned_to,dtheta_arcsec,comment\n')
                F.write(f'{cat[m1]["OBJNAME"]},{cat[m1]["PGC"]},{cat[m2]["OBJNAME"]},{cat[m2]["PGC"]},{dtheta_ned:.3f},{cat[m1]["OBJTYPE"]}\n')
                if verbose:
                    log.info(f'Writing {cat[m1]["OBJNAME"]} (PGC {cat[m1]["PGC"]}) --> {cat[m2]["OBJNAME"]} (PGC {cat[m2]["PGC"]})')

                # Also check to see if the "m2" source similarly matches
                # another object in the catalog.
                #_, m3, _ = match_radec(cat[m2]['RA_HYPERLEDA'], cat[m2]['DEC_HYPERLEDA'], cat['RA'], cat['DEC'], 1.5/3600.)
                #m3 = m3[~np.isin(m3, [m1, m2])] # remove self
                if len(m3) == 0: # no match
                    continue
                m3 = m3[0]

                # delta-theta between NED (old) and HyperLeda (adopted)
                dtheta_ned = arcsec_between(cat[m2]['RA'], cat[m2]['DEC'], cat[m2]['RA_HYPERLEDA'], cat[m2]['DEC_HYPERLEDA'])

                # objname_ned_from,objname_ned_to,dtheta_arcsec,comment\n')
                F.write(f'{cat[m2]["OBJNAME"]},{cat[m2]["PGC"]},{cat[m3]["OBJNAME"]},{cat[m3]["PGC"]},{dtheta_ned:.3f},{cat[m2]["OBJTYPE"]}\n')
                if verbose:
                    log.info(f'  Adding {cat[m2]["OBJNAME"]} (PGC {cat[m2]["PGC"]}) --> {cat[m3]["OBJNAME"]} (PGC {cat[m3]["PGC"]})')

        F.close()

    # Update the input catalog.
    crossids = Table.read(crossidfile, format='csv', comment='#')
    log.info(f'Read {len(crossids):,d} rows from {crossidfile}')

    newcat = fullcat.copy()

    obj_from = crossids['objname_from'].value
    obj_to = crossids['objname_to'].value
    for obj in (obj_from, obj_to):
        I = np.isin(newcat['OBJNAME'].value, obj)
        if np.sum(I) != len(obj):
            raise ValueError('Some objects not found!')
            #log.info(obj[~np.isin(obj, newcat['OBJNAME'][I].value)])

    drop, dropcat = [], []
    for crossid in crossids:
        m1 = np.where(crossid['objname_from'] == newcat['OBJNAME'])[0][0]
        m2 = np.where(crossid['objname_to']== newcat['OBJNAME'])[0][0]
        drop.append(m1)
        dropcat.append(newcat[m1][cols])
        if verbose:
            log.info(f'Copying {newcat[m1]["OBJNAME"]} (PGC {newcat[m1]["PGC"]}, {newcat[m1]["OBJTYPE"]}) to ' + \
                  f'{newcat[m2]["OBJNAME"]} (PGC {newcat[m2]["PGC"]}, PGC {newcat[m2]["OBJTYPE"]})')
        for col in ['OBJNAME_HYPERLEDA', 'RA_HYPERLEDA', 'DEC_HYPERLEDA', 'DIAM_LIT', 'BA_LIT', 'PA_LIT',
                    'DIAM_HYPERLEDA', 'BA_HYPERLEDA', 'PA_HYPERLEDA', 'ROW_HYPERLEDA', 'PGC']:
            new = newcat[col][m1]
            old = newcat[col][m2]
            if new == '' and old == '':
                raise ValueError('Special case - write me')
            if 'DIAM_LIT' in col:
                new = np.max((new, old))
            if (new != '' or new != -99) and (old == '' or old == -99):
                if verbose:
                    log.info(f'  Replacing {col}: {old} --> {new}')
                # Do not create duplicate PGC or coordinate values...
                newcat[col][m2] = new
                newcat[col][m1] = old
            else:
                if verbose:
                    log.info(f'  Keeping {col}: {old} (ignoring {new})')
        for col in ['RA', 'DEC']:
            old = newcat[col][m2]
            new = newcat[f'{col}_HYPERLEDA'][m2]
            if verbose:
                log.info(f'  Replacing {col}: {old} --> {new}')
            newcat[col][m1] = new
        if verbose:
            print()

    drop = np.hstack(drop)
    dropcat = vstack(dropcat)

    log.info(f'Dropping {len(drop):,d} cross-id errors (all GTrpl and GPair) from the catalog.')
    newcat = newcat[np.delete(np.arange(len(newcat)), drop)]

    # Read and act on the "VI actions" file.
    actionsfile = resources.files('SGA').joinpath('data/SGA2025/SGA2025-vi-actions.csv')
    actions = Table.read(actionsfile, format='csv', comment='#')

    for action in np.unique(actions['action']):
        match action:
            # drop from the sample
            case 'drop':
                obj = actions[action == actions['action']]['objname'].value
                uobj, cc = np.unique(obj, return_counts=True)
                if np.max(cc) != 1:
                    log.info('Warning: duplicates in actions file can cause problems!')
                    log.info(uobj[cc > 1])
                    raise ValueError()
                I = np.isin(newcat['OBJNAME'].value, obj)
                if np.sum(I) != len(obj):
                    #for oo in obj:
                    #    if np.sum(np.isin(newcat['OBJNAME'], oo)) != 1:
                    #        log.info(oo)
                    log.info('The following objects are in the actions file but do not exist!')
                    log.info(obj[~np.isin(obj, newcat['OBJNAME'][I].value)])
                    raise ValueError()
                if verbose:
                    log.info(f'Action: dropping {len(obj):,d} objects.')
                newcat = newcat[~I]
            # NED coordinates are wrong; adopt HyperLeda
            case 'hyperleda-coords':
                obj = actions[action == actions['action']]['objname'].value
                I = np.isin(newcat['OBJNAME'].value, obj)
                if np.sum(I) != len(obj):
                    log.info('The following objects are in the actions file but do not exist!')
                    log.info(obj[~np.isin(obj, newcat['OBJNAME'][I].value)])
                    raise ValueError()
                if verbose:
                    log.info(f'Action: adopting HyperLeda coordinates for {len(obj):,d} object(s).')
                for col in ['RA', 'DEC']:
                    newcat[col][I] = newcat[I][f'{col}_HYPERLEDA']
            case _:
                log.info(f'Unrecognized action {action}!')
                raise ValueError


    # resolve close sources / duplicates (but only in the cross-ID fields)
    match_new = match_radec(dropcat['RA'].value, dropcat['DEC'].value, newcat['RA'].value,
                            newcat['DEC'].value, 75./3600., indexlist=True, notself=False)
    match_new = [mm for mm in match_new if mm is not None]

    I = np.unique(np.hstack(match_new))
    dups = resolve_close(newcat[I], newcat[I], maxsep=1., allow_vetos=True, verbose=verbose, trim=False)
    I = np.isin(newcat['ROW_PARENT'], dups[dups['PRIMARY'] == False]['ROW_PARENT'])
    log.info(f'Dropping {np.sum(I):,d} close pairs.')
    newcat = newcat[~I]

    # sort by diameter
    srt = np.argsort(np.max((dropcat['DIAM_LIT'].value, dropcat['DIAM_HYPERLEDA'].value), axis=0))[::-1]
    dropcat = dropcat[srt]

    # add a VI bit; all these systems have been visually checked
    width_arcsec = 75.
    matches = match_radec(dropcat['RA'].value, dropcat['DEC'].value, newcat['RA'].value,
                          newcat['DEC'].value, width_arcsec/3600., indexlist=True, notself=False)
    matches = [mm for mm in matches if mm is not None]
    matches = np.unique(np.hstack(matches))
    #newcat['VI'] = np.zeros(len(newcat), bool)
    #newcat['VI'][matches] = True

    # Build QA showing the sources at the center of each of the objects dropped
    # (in "dropcat").
    if build_qa:
        jpgdir = os.path.join(sga_dir(), 'parent', 'vi', f'crossid-errors-viewer')
        pngdir = os.path.join(sga_dir(), 'parent', 'vi', f'crossid-errors-png')
        pdfdir = os.path.join(sga_dir(), 'parent', 'vi', f'crossid-errors-pdf')
        if not os.path.isdir(pdfdir):
            os.makedirs(pdfdir)

        # make multiple pdfs
        nperpdf = 192
        npdf = int(np.ceil(len(crossids) / nperpdf))

        for ii in range(npdf):
            ss = ii * nperpdf
            ee = (ii + 1) * nperpdf
            if ee > len(dropcat)-1:
                ee = len(dropcat)-1
            pdffile = os.path.join(pdfdir, f'vi-crossid-errors-{ss:04}-{ee-1:04}.pdf')

            #multipage_skypatch(dropcat[ss:ee], cat=fullcat, pngsuffix='group', jpgdir=jpgdir,
            multipage_skypatch(dropcat[ss:ee], cat=newcat, width_arcsec=width_arcsec, clip=True,
                               pngsuffix='group', jpgdir=jpgdir, pngdir=pngdir, pdffile=pdffile,
                               verbose=verbose, overwrite=True, add_title=True, cleanup=cleanup)

    #newcat[match_new[np.where(dropcat['OBJNAME'] == 'CGCG 039-044')[0][0]]][cols]
    #newcat[np.flatnonzero(np.char.find(newcat['OBJNAME'].value, 'APMUKS') != -1)][cols]

    #m1, m2, _ = match_radec(newcat['RA'], newcat['DEC'], 0.3728, 13.0985, 75./3600.)
    #m1, m2, _ = match_radec(newcat['RA'], newcat['DEC'], newcat[newcat['PGC'] == 680515]['RA'], newcat[newcat['PGC'] == 680515]['DEC'], 120./3600.)
    #m1, m2, _ = match_radec(newcat['RA'], newcat['DEC'], newcat[newcat['PGC'] == 680515]['RA'], newcat[newcat['PGC'] == 680515]['DEC'], 120./3600.)
    #newcat[m1][cols]
    #qa_skypatch(newcat[m1[15]], group=newcat[m1], pngdir='.', jpgdir='.', verbose=True, overwrite=True, width_arcmin=4.)

    #prefix = np.array(list(zip(*np.char.split(newcat['OBJNAME'].value, ' ').tolist()))[0])
    #allprefix = np.array(list(zip(*np.char.split(fullcat['OBJNAME'].value, ' ').tolist()))[0])
    #newcat[match_new[np.where(dropcat['OBJNAME'] == 'ESO 292-IG 010')[0][0]]][cols]
    #fullcat[match_full[np.where(dropcat['OBJNAME'] == 'APMUKS(BJ) B020550.65-13135 ID')[0][0]]][cols]
    #newcat[np.flatnonzero(np.char.find(newcat['OBJNAME'].value, 'APMUKS') != -1)][cols]

    return newcat


def build_parent_nocuts(verbose=True):
    """Merge the external catalogs from SGA2025-query-ned.

    """
    import re
    from astropy.table import join
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    import astropy.units as u
    from SGA.io import (read_hyperleda, version_hyperleda, nedfriendly_hyperleda,
                        read_hyperleda_galaxies, version_hyperleda_galaxies,
                        read_hyperleda_multiples, version_hyperleda_multiples,
                        read_hyperleda_noobjtype, version_hyperleda_noobjtype,
                        read_nedlvs, version_nedlvs, read_sga2020, read_lvd,
                        version_lvd, nedfriendly_lvd, version_custom_external,
                        read_custom_external)

    def readit(catalog, version, bycoord=False):
        if bycoord:
            suffix = 'bycoord'
        else:
            suffix = 'byname'
        datafile = os.path.join(sga_dir(), 'parent', 'external', f'NED{suffix}-{catalog}_{version}.fits')
        data = Table(fitsio.read(datafile))
        log.info(f'Read {len(data):,d} objects from {datafile}')
        return data


    def populate_parent(input_cat, input_basic, verbose=False):
        from SGA.io import parent_datamodel
        parent = parent_datamodel(len(input_cat))
        for col in parent.columns:
            if col in input_cat.columns:
                if verbose:
                    log.info(f'Populating {col}')
                parent[col] = input_cat[col]
            if col in input_basic.columns:
                if verbose:
                    log.info(f'Populating {col}')
                parent[col] = input_basic[col]
        return parent


    log.info('#####')
    log.info('Input data:')
    custom = read_custom_external(overwrite=True) # always regenerate the FITS file
    lvd = read_lvd()
    nedlvs = read_nedlvs()
    sga2020 = read_sga2020()
    hyper = read_hyperleda()
    print()

    ned_lvd = readit('LVD', version_lvd())
    ned_nedlvs = readit('NEDLVS', version_nedlvs())

    ## For Kim!!
    #basic = get_basic_geometry(ned_nedlvs, galaxy_column='OBJNAME')
    #basic.write('/Users/ioannis/Downloads/NEDgeometry-NEDLVS_20210922_v2.fits', overwrite=True)

    # merge all the ned_hyper catalogs and reset row (to match SGA.io.read_hyperleda())
    ned_hyper_galaxies = readit('HyperLeda-galaxies', version_hyperleda_galaxies())
    ned_hyper_multiples = readit('HyperLeda-multiples', version_hyperleda_multiples())
    ned_hyper_noobjtype = readit('HyperLeda-noobjtype', version_hyperleda_noobjtype())
    ned_hyper_galaxies_coords = readit('HyperLeda-galaxies-coords', f'{version_hyperleda_galaxies()}', bycoord=True)
    ned_hyper_multiples_coords = readit('HyperLeda-multiples-coords', f'{version_hyperleda_multiples()}', bycoord=True)
    ned_hyper_noobjtype_coords = readit('HyperLeda-noobjtype-coords', f'{version_hyperleda_noobjtype()}', bycoord=True)
    ned_hyper_galaxies_coords.remove_columns(['INPUT_POSITION', 'INPUT_RA', 'INPUT_DEC', 'SEPARATION'])
    ned_hyper_multiples_coords.remove_columns(['INPUT_POSITION', 'INPUT_RA', 'INPUT_DEC', 'SEPARATION'])
    ned_hyper_noobjtype_coords.remove_columns(['INPUT_POSITION', 'INPUT_RA', 'INPUT_DEC', 'SEPARATION'])

    ned_hyper_galaxies = vstack((ned_hyper_galaxies, ned_hyper_galaxies_coords))
    indx_hyper, indx_galaxies = match(hyper['OBJNAME'], ned_hyper_galaxies['OBJNAME'])
    #indx_hyper, indx_galaxies = match(hyper['ROW_GALAXIES'], ned_hyper_galaxies['ROW'])
    ned_hyper_galaxies['ROW'][indx_galaxies] = hyper[indx_hyper]['ROW']

    ned_hyper_multiples = vstack((ned_hyper_multiples, ned_hyper_multiples_coords))
    indx_hyper, indx_multiples = match(hyper['OBJNAME'], ned_hyper_multiples['OBJNAME'])
    #indx_hyper, indx_multiples = match(hyper['ROW_MULTIPLES'], ned_hyper_multiples['ROW'])
    ned_hyper_multiples['ROW'][indx_multiples] = hyper[indx_hyper]['ROW']

    ned_hyper_noobjtype = vstack((ned_hyper_noobjtype, ned_hyper_noobjtype_coords))
    indx_hyper, indx_noobjtype = match(hyper['OBJNAME'], ned_hyper_noobjtype['OBJNAME'])
    #indx_hyper, indx_noobjtype = match(hyper['ROW_NOOBJTYPE'], ned_hyper_noobjtype['ROW'])
    ned_hyper_noobjtype['ROW'][indx_noobjtype] = hyper[indx_hyper]['ROW']

    ned_hyper = vstack((ned_hyper_galaxies, ned_hyper_multiples, ned_hyper_noobjtype))
    ned_hyper = ned_hyper[np.argsort(ned_hyper['ROW'])]
    #match_radec(ned_hyper['RA'], ned_hyper['DEC'], custom['RA'], custom['DEC'], 3./3600.)

    nobj_ned_lvd = len(ned_lvd)
    nobj_ned_nedlvs = len(ned_nedlvs)
    nobj_ned_hyper = len(ned_hyper)
    hyper_noned = hyper[~np.isin(hyper['ROW'], ned_hyper['ROW'])]

    # [0] Preprocess the data.

    # ned_nedlvs - 21 objects are duplicates, apparently because of
    # cross-identification problems in NED. Keep just the first one of each
    # occurrance here (=10 unique objects); plus add 8 more duplicates discovered "by hand"
    print()
    log.info('#####')
    log.info('ned_nedlvs:')
    #dups = ['Andromeda XXXIII', 'SDSS J002041.45+083701.2', 'SDSS J104653.19+124441.4']
    #nedlvs = nedlvs[~np.isin(nedlvs['OBJNAME'], dups)]
    #ned_nedlvs = ned_nedlvs[~np.isin(ned_nedlvs['OBJNAME'], dups)]

    #dups = ['Andromeda XXXIII', 'PGC1 5067061 NED001', 'HIPASS J0021+08', 'SDSS J002041.45+083701.2', 'SDSS J095549.64+691957.4', 'SDSS J141708.23+134105.7', 'SDSS J104653.19+124441.4', 'Leo dw A']
    ##nedlvs[np.isin(nedlvs['ROW'], [1869664, 1835853])]
    ##nedlvs[np.isin(nedlvs['ROW'], [1589395, 1827949])]
    ##nedlvs[np.isin(nedlvs['ROW'], [1742809, 328053])]
    ##nedlvs[np.isin(nedlvs['ROW'], [1656036, 1430675])]
    #jj = nedlvs[np.isin(nedlvs['OBJNAME'], dups)]
    #jj[np.argsort(jj['RA'])]

    col = 'OBJNAME_NED'
    rr, cc = np.unique(ned_nedlvs[col], return_counts=True)
    dups = rr[cc>1].value
    #dups = np.hstack((rr[cc>1].value, ['SDSS J141708.23+134105.7', 'Leo dw A', 'HIPASS J0021+08',
    #                                   'SDSS J104653.19+124441.4', 'SDSS J095549.64+691957.4',
    #                                   'SDSS J002041.45+083701.2', 'PGC1 5067061 NED001',
    #                                   'Andromeda XXXIII']))
    dd = ned_nedlvs[np.isin(ned_nedlvs[col], dups)]
    dd = dd[np.argsort(dd[col])]

    basic_dd = get_basic_geometry(dd, galaxy_column='OBJNAME_NED', verbose=verbose)
    #basic_dd.rename_column('GALAXY', 'OBJNAME')
    #basic_dd['RA'] = dd['RA']
    #basic_dd['DEC'] = dd['DEC']
    #basic_dd['Z'] = dd['Z']
    #basic_dd['OBJTYPE'] = dd['OBJTYPE']
    #basic_dd['DIAM_HYPERLEDA'] = -99.
    #basic_dd['PGC'] = -99
    #res = resolve_close(ned_nedlvs, basic_dd, maxsep=3.1, objname_column='OBJNAME', keep_all_mergers=False, verbose=True, trim=False)
    #I = np.where(res['PRIMARY'])[0]
    toss = []
    for objname in np.unique(dd[col]):
        I = np.where(ned_nedlvs[col] == objname)[0]
        J = np.where(basic_dd['GALAXY'] == objname)[0]
        this = np.where(basic_dd[J]['DIAM_LIT'] > 0.)[0]
        if len(this) == 0:
            toss.append(np.delete(I, np.argmin(ned_nedlvs[I]['ROW'])))
        else:
            this = this[np.argsort(ned_nedlvs[I][this]['ROW'])]
            toss.append(np.delete(I, this[0]))
        #I = I[np.argsort(ned_nedlvs[col][I])]
        #toss.append(I[1:]) # keep the zeroth match
    toss = np.hstack(toss)
    log.info(f'Removing {len(toss):,d}/{len(ned_nedlvs):,d} ' + \
             f'({100.*len(toss)/len(ned_nedlvs):.1f}%) {col} duplicates.')
    ned_nedlvs = ned_nedlvs[np.delete(np.arange(len(ned_nedlvs)), toss)]

    c_nedlvs = SkyCoord(ra=ned_nedlvs['RA']*u.deg, dec=ned_nedlvs['DEC']*u.deg)
    indx_nedlvs, sep2d, _ = match_coordinates_sky(
        c_nedlvs, c_nedlvs, nthneighbor=2)
    dd = ned_nedlvs[sep2d.arcsec == 0.]
    dd = dd[np.argsort(dd['RA'])]
    basic_dd = get_basic_geometry(dd, galaxy_column='OBJNAME_NED',
                                  verbose=verbose)
    radecs = np.char.add(np.round(dd['RA'], 10).astype(str),
                         np.round(dd['DEC'], 10).astype(str))
    ref_radecs = np.char.add(np.round(ned_nedlvs['RA'], 10).astype(str),
                             np.round(ned_nedlvs['DEC'], 10).astype(str))
    toss = []
    for radec in np.unique(radecs):
        I = np.where(radec == ref_radecs)[0]
        J = np.where(radec == radecs)[0]
        this = np.where(basic_dd[J]['DIAM_LIT'] > 0.)[0]
        if len(this) == 0:
            toss.append(np.delete(I, np.argmin(ned_nedlvs[I]['ROW'])))
        else:
            this = this[np.argsort(ned_nedlvs[I][this]['ROW'])]
            toss.append(np.delete(I, this[0]))
    toss = np.hstack(toss)
    log.info(f'Removing {len(toss):,d}/{len(ned_nedlvs):,d} ' + \
             f'({100.*len(toss)/len(ned_nedlvs):.1f}%) coordinate duplicates.')
    ned_nedlvs = ned_nedlvs[np.delete(np.arange(len(ned_nedlvs)), toss)]

    # Toss out non-galaxies in ned_nedlvs. But note:
    # * Need to make sure the individual members of the GGroup systems are
    #   in the final parent sample.
    # https://ned.ipac.caltech.edu/help/ui/nearposn-list_objecttypes?popup=1
    toss = np.where(np.isin(ned_nedlvs['OBJTYPE'], ['QSO', 'Q_Lens', 'G_Lens', '*',
                                                    'Other', 'GGroup'#,
                                                    #'GPair', 'GTrpl'
                                                    ]))[0]
    log.info(f'Removing {len(toss):,d}/{len(ned_nedlvs):,d} ' + \
             f'({100.*len(toss)/len(ned_nedlvs):.1f}%) non-galaxies.')
    ned_nedlvs = ned_nedlvs[np.delete(np.arange(len(ned_nedlvs)), toss)]

    ## ned_hyper - 1 object (WINGSJ125256.27-152110.4) is a duplicate. As
    ## the primary object, it's PGC4777821, but as the alternate object,
    ## it's also [CZ2003]1631C-0295:095 = PGC6729485. In addition, remove
    ## the ~2500 objects not in NED and the ~11k objects resolve to the same
    ## object in NED; choose the first one.
    #warn = np.array(['WARNING' in objnote for objnote in ned_hyper['OBJECT_NOTE']])
    #log.info(f'Removing {np.sum(warn):,d}/{len(ned_hyper):,d} objects with NED warnings from ned_hyper.')
    #ned_hyper = ned_hyper[~warn]

    #col = 'OBJNAME'
    #rr, cc = np.unique(ned_hyper[col], return_counts=True)
    #dd = ned_hyper[np.isin(ned_hyper[col], rr[cc>1].value)]
    #dd = dd[np.argsort(dd[col])]
    #toss = []
    #for objname in np.unique(dd[col]):
    #    I = np.where(ned_hyper[col] == objname)[0]
    #    I = I[np.argsort(ned_hyper[col][I])]
    #    toss.append(I[1:]) # keep the zeroth match
    #toss = np.hstack(toss)
    #log.info(f'Removing {len(toss):,d}/{len(ned_hyper):,d} {col} duplicates from ned_hyper.')
    #ned_hyper = ned_hyper[np.delete(np.arange(len(ned_hyper)), toss)]
    print()
    log.info('ned_hyper:')

    col = 'OBJNAME_NED'
    rr, cc = np.unique(ned_hyper[col], return_counts=True)
    dd = ned_hyper[np.isin(ned_hyper[col], rr[cc>1].value)]
    dd = dd[np.argsort(dd[col])]
    basic_dd = get_basic_geometry(dd, galaxy_column='OBJNAME_NED',
                                  verbose=verbose)
    toss = []
    for objname in np.unique(dd[col]):
        I = np.where(ned_hyper[col] == objname)[0]
        J = np.where(basic_dd['GALAXY'] == objname)[0]
        this = np.where(basic_dd[J]['DIAM_LIT'] > 0.)[0]
        if len(this) == 0:
            toss.append(np.delete(I, np.argmin(ned_hyper[I]['ROW'])))
        else:
            this = this[np.argsort(ned_hyper[I][this]['ROW'])]
            toss.append(np.delete(I, this[0]))
    toss = np.hstack(toss)
    log.info(f'Removing {len(toss):,d}/{len(ned_hyper):,d} ' + \
             f'({100.*len(toss)/len(ned_hyper):.1f}%) {col} duplicates.')
    ned_hyper = ned_hyper[np.delete(np.arange(len(ned_hyper)), toss)]

    c_hyper = SkyCoord(ra=ned_hyper['RA']*u.deg, dec=ned_hyper['DEC']*u.deg)
    indx_hyper, sep2d, _ = match_coordinates_sky(c_hyper, c_hyper,
                                                 nthneighbor=2)
    dd = ned_hyper[sep2d.arcsec == 0.]
    dd = dd[np.lexsort((dd['PGC'], dd['RA']))]
    basic_dd = get_basic_geometry(dd, galaxy_column='OBJNAME_NED',
                                  verbose=verbose)

    basic_dd.rename_column('GALAXY', 'OBJNAME')
    basic_dd['RA'] = dd['RA']
    basic_dd['DEC'] = dd['DEC']
    basic_dd['Z'] = dd['Z']
    basic_dd['OBJTYPE'] = dd['OBJTYPE']
    basic_dd['DIAM_HYPERLEDA'] = -99.
    basic_dd['PGC'] = -99
    res = resolve_close(basic_dd, basic_dd, maxsep=1., objname_column='OBJNAME',
                        keep_all_mergers=False, verbose=False, trim=False)
    toss = np.where(np.isin(ned_hyper['ROW'], dd[~res['PRIMARY']]['ROW']))[0]

    #radecs = np.char.add(np.round(dd['RA'], 10).astype(str), np.round(dd['DEC'], 10).astype(str))
    #ref_radecs = np.char.add(np.round(ned_hyper['RA'], 10).astype(str), np.round(ned_hyper['DEC'], 10).astype(str))
    #toss = []
    #for radec in np.unique(radecs):
    #    I = np.where(radec == ref_radecs)[0]
    #    J = np.where(radec == radecs)[0]
    #    this = np.where(basic_dd[J]['DIAM_LIT'] > 0.)[0]
    #    if len(this) == 0:
    #        toss.append(np.delete(I, np.argmin(ned_hyper[I]['ROW'])))
#   #     elif len(this) == 1:
#her#e
#   #         this = this[np.argsort(ned_hyper[I][this]['ROW'])]
    #    else:
    #        this = this[np.argsort(ned_hyper[I][this]['ROW'])]
    #        toss.append(np.delete(I, this[0]))
    #toss = np.hstack(toss)
    log.info(f'Removing {len(toss):,d}/{len(ned_hyper):,d} ({100.*len(toss)/len(ned_hyper):.1f}%) coordinate duplicates.')
    ned_hyper = ned_hyper[np.delete(np.arange(len(ned_hyper)), toss)]

    # Toss out non-galaxies in ned_hyper. But note:
    # * Need to make sure the individual members of the GGroup systems are
    #   in the final parent sample.
    # * Some objects classified as point sources (*) have SDSS redshifts,
    #   so the classification is wrong (e.g., GAMA743045=SDSSJ141614.97-005648.2)
    # * Also throw out VIRGO01, which incorrectly maps to 'Virgo I Dwarf'.

    # https://ned.ipac.caltech.edu/help/ui/nearposn-list_objecttypes?popup=1
    toss = np.where(np.isin(ned_hyper['OBJTYPE'], ['PofG', '!V*', '!PN', '**', 'GClstr', 'WD*',
                                                   'Red*', '!HII', 'C*', 'PN', '*Ass', 'Blue*',
                                                   '!**', 'SN', '!*', 'Other', 'SNR', '*Cl',
                                                   '!WD*', 'GGroup', 'WR*',
                                                   #'GPair', 'GTrpl',
                                                   'V*', '*', 'HII', 'Nova', 'Neb', 'RfN', '!V*', '!C*',
                                                   'QSO', 'Q_Lens', 'G_Lens']))[0]
    toss = np.hstack((toss, np.where(ned_hyper['OBJNAME'] == 'VIRGO01')[0]))
    log.info(f'Removing {len(toss):,d}/{len(ned_hyper):,d} ({100.*len(toss)/len(ned_hyper):.1f}%) non-galaxies.')
    ned_hyper = ned_hyper[np.delete(np.arange(len(ned_hyper)), toss)]

    # check
    print()
    log.info('After basic cuts:')
    for name, cat, norig in zip(['ned_lvd', 'ned_nedlvs', 'ned_hyper'],
                                [ned_lvd, ned_nedlvs, ned_hyper],
                                [nobj_ned_lvd, nobj_ned_nedlvs, nobj_ned_hyper]):
        nobj = len(cat)
        log.info(f'{name}: {nobj:,d}/{norig:,d} objects')
        for col in ['OBJNAME', 'OBJNAME_NED', 'ROW']:
            assert(len(np.unique(cat[col])) == nobj)
            #rr, cc = np.unique(cat[col], return_counts=True)
            ##log.info(rr[cc>1])
            #bb = cat[np.isin(cat[col], rr[cc>1].value)]
            #bb = bb[np.argsort(bb[col])]

    # [1] - Match HyperLeda{-altname} to NEDLVS using OBJNAME_NED.
    print()
    log.info('#####')

    keys = np.array(ned_nedlvs.colnames)
    keys = keys[~np.isin(keys, ['OBJNAME', 'ROW', 'RA', 'DEC'])]
    out1 = join(ned_hyper, ned_nedlvs, keys=keys, table_names=['HYPERLEDA', 'NEDLVS'])

    # round-off
    out1.rename_columns(['RA_HYPERLEDA', 'DEC_HYPERLEDA'], ['RA', 'DEC'])
    out1.remove_columns(['RA_NEDLVS', 'DEC_NEDLVS'])
    out1.rename_columns(['RA', 'DEC', 'Z'], ['RA_NED', 'DEC_NED', 'Z_NED'])
    log.info(f'Matched {len(out1):,d}/{len(ned_hyper):,d} ({100.*len(out1)/len(ned_hyper):.1f}%) ned_hyper and ' + \
          f'{len(out1):,d}/{len(ned_nedlvs):,d} ({100.*len(out1)/len(ned_nedlvs):.1f}%) ned_nedlvs objects using OBJNAME_NED.')

    basic_out1 = get_basic_geometry(out1, galaxy_column='OBJNAME_NED', verbose=verbose)

    #indx_out, indx_hyper = match(out1['ROW_HYPERLEDA'], hyper['ROW'])
    #out1['OBJNAME_HYPERLEDA'][indx_out] = hyper['OBJNAME'][indx_hyper]
    #out1 = out1[np.argsort(out1['ROW_HYPERLEDA'])]

    parent1 = populate_parent(out1, basic_out1, verbose=verbose)

    indx_parent, indx_hyper = match(parent1['ROW_HYPERLEDA'], hyper['ROW'])
    if verbose:
        for col in ['RA_HYPERLEDA', 'DEC_HYPERLEDA', 'Z_HYPERLEDA']:
            log.info(f'Populating {col}')
    parent1['RA_HYPERLEDA'][indx_parent] = hyper['RA'][indx_hyper]
    parent1['DEC_HYPERLEDA'][indx_parent] = hyper['DEC'][indx_hyper]
    I = np.where(~np.isnan(hyper['V'][indx_hyper]))[0]
    parent1['Z_HYPERLEDA'][indx_parent[I]] = hyper['V'][indx_hyper[I]] / 2.99e5

    indx_parent, indx_nedlvs = match(parent1['ROW_NEDLVS'], nedlvs['ROW'])
    if verbose:
        for col in ['RA_NEDLVS', 'DEC_NEDLVS', 'Z_NEDLVS']:
            log.info(f'Populating {col}')
    parent1['RA_NEDLVS'][indx_parent] = nedlvs['RA'][indx_nedlvs]
    parent1['DEC_NEDLVS'][indx_parent] = nedlvs['DEC'][indx_nedlvs]
    I = np.where(~np.isnan(nedlvs['Z'][indx_nedlvs]))[0]
    parent1['Z_NEDLVS'][indx_parent[I]] = nedlvs['Z'][indx_nedlvs[I]]

    for col in ['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_NEDLVS', 'ROW_HYPERLEDA', 'ROW_NEDLVS']:
        assert(len(np.unique(parent1[col])) == len(parent1))
    print()
    log.info(f'Parent 1: N={len(parent1):,d}')

    # [2] - Add as many of the remaining ned_hyper objects as possible. Special
    # case VIRGO1, which incorrectly matches (in NED) to 'Virgo I Dwarf' rather
    # than 'Virgo I'.
    print()
    log.info('#####')
    miss_hyper = ned_hyper[np.logical_and(~np.isin(ned_hyper['ROW'], parent1['ROW_HYPERLEDA']),
                                          (ned_hyper['OBJNAME'] != 'VIRGO1'))]
    miss_hyper.rename_columns(['OBJNAME', 'ROW'], ['OBJNAME_HYPERLEDA', 'ROW_HYPERLEDA'])
    miss_hyper.rename_columns(['RA', 'DEC', 'Z'], ['RA_NED', 'DEC_NED', 'Z_NED'])

    log.info(f'Adding the remaining {len(miss_hyper):,d} objects from ned_hyper which did not name-match ned_nedlvs.')
    basic_miss_hyper = get_basic_geometry(miss_hyper, galaxy_column='OBJNAME_NED', verbose=verbose)

    parent2 = populate_parent(miss_hyper, basic_miss_hyper, verbose=verbose)

    indx_parent, indx_hyper = match(parent2['ROW_HYPERLEDA'], hyper['ROW'])
    if verbose:
        for col in ['RA_HYPERLEDA', 'DEC_HYPERLEDA', 'Z_HYPERLEDA']:
            log.info(f'Populating {col}')
    parent2['RA_HYPERLEDA'][indx_parent] = hyper['RA'][indx_hyper]
    parent2['DEC_HYPERLEDA'][indx_parent] = hyper['DEC'][indx_hyper]
    I = np.where(~np.isnan(hyper['V'][indx_hyper]))[0]
    parent2['Z_HYPERLEDA'][indx_parent[I]] = hyper['V'][indx_hyper[I]] / 2.99e5

    for col in ['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'ROW_HYPERLEDA']:
        assert(len(np.unique(parent2[col])) == len(parent2))
    print()
    log.info(f'Parent 2: N={len(parent2):,d}')
    # [3] - Add the rest of the ned_nedlvs objects, being careful about exact
    # duplicates.
    print()
    log.info('#####')

    parent = vstack((parent1, parent2))

    miss_nedlvs = ned_nedlvs[~np.isin(ned_nedlvs['ROW'], parent['ROW_NEDLVS'])]
    miss_nedlvs.rename_columns(['OBJNAME', 'ROW'], ['OBJNAME_NEDLVS', 'ROW_NEDLVS'])
    miss_nedlvs.rename_columns(['RA', 'DEC', 'Z'], ['RA_NED', 'DEC_NED', 'Z_NED'])
    log.info(f'Analyzing the remaining {len(miss_nedlvs):,d} ned_nedlvs objects.')

    c_parent = SkyCoord(ra=parent['RA_NED']*u.deg, dec=parent['DEC_NED']*u.deg)
    c_nedlvs = SkyCoord(ra=miss_nedlvs['RA_NED']*u.deg, dec=miss_nedlvs['DEC_NED']*u.deg)
    indx_dup_nedlvs, sep2d, _ = c_parent.match_to_catalog_sky(c_nedlvs)
    indx_dup_parent = np.where(sep2d.arcsec == 0.)[0]
    indx_dup_nedlvs = indx_dup_nedlvs[indx_dup_parent]

    # also remove some duplicates that arose because my NEDLVS query came well
    # before (~July 2025) my HyperLeda-multiples query, and I guess some of the
    # coordinates changed...
    _miss_nedlvs = miss_nedlvs[np.delete(np.arange(len(miss_nedlvs)), indx_dup_nedlvs)]
    dups = _miss_nedlvs[np.isin(_miss_nedlvs['OBJNAME_NED'], parent['OBJNAME_NED'])]['OBJNAME_NED'].value
    indx_dup_nedlvs = np.hstack((indx_dup_nedlvs, np.where(np.isin(miss_nedlvs['OBJNAME_NED'], dups))[0]))

    #dup_parent = parent[indx_dup_parent]
    #dup_parent['OBJNAME_HYPERLEDA', 'OBJNAME_NED', 'OBJNAME_NEDLVS', 'RA_NED', 'DEC_NED'][:10]
    #miss_nedlvs[indx_dup_nedlvs]['OBJNAME_NEDLVS', 'OBJNAME_NED', 'RA_NED', 'DEC_NED'][:10]

    log.info(f'Removing {len(indx_dup_nedlvs):,d}/{len(miss_nedlvs):,d} ({100.*len(indx_dup_nedlvs)/len(miss_nedlvs):.1f}%) ' + \
          f'ned_nedlvs duplicates (sep=0.0 arcsec) already in parent sample.')
    #parent = parent[np.delete(np.arange(len(parent)), indx_dup_parent)]
    miss_nedlvs = miss_nedlvs[np.delete(np.arange(len(miss_nedlvs)), indx_dup_nedlvs)]

    basic_miss_nedlvs = get_basic_geometry(miss_nedlvs, galaxy_column='OBJNAME_NED', verbose=verbose)

    parent3 = populate_parent(miss_nedlvs, basic_miss_nedlvs, verbose=verbose)

    indx_parent, indx_nedlvs = match(parent3['ROW_NEDLVS'], nedlvs['ROW'])
    if verbose:
        for col in ['RA_NEDLVS', 'DEC_NEDLVS', 'Z_NEDLVS']:
            log.info(f'Populating {col}')
    parent3['RA_NEDLVS'][indx_parent] = nedlvs['RA'][indx_nedlvs]
    parent3['DEC_NEDLVS'][indx_parent] = nedlvs['DEC'][indx_nedlvs]
    I = np.where(~np.isnan(nedlvs['Z'][indx_nedlvs]))[0]
    parent3['Z_NEDLVS'][indx_parent[I]] = nedlvs['Z'][indx_nedlvs[I]]

    for col in ['OBJNAME_NED', 'OBJNAME_NEDLVS', 'ROW_NEDLVS']:
        assert(len(np.unique(parent3[col])) == len(parent3))
    print()
    log.info(f'Parent 3: N={len(parent3):,d}')
    parent = vstack((parent, parent3))

    # [4] - Add any outstanding hyper objects with measured diameters.
    print()
    log.info('#####')

    miss_hyper = hyper_noned[~np.isin(hyper_noned['ROW'], parent['ROW_HYPERLEDA'])]
    miss_hyper.rename_columns(['OBJNAME', 'ROW'], ['OBJNAME_HYPERLEDA', 'ROW_HYPERLEDA'])
    miss_hyper.rename_columns(['RA', 'DEC'], ['RA_HYPERLEDA', 'DEC_HYPERLEDA'])
    miss_hyper['Z_HYPERLEDA'] = np.zeros(len(miss_hyper)) - 99.
    I = np.where(~np.isnan(miss_hyper['V']))[0]
    miss_hyper['Z_HYPERLEDA'][I] = hyper['V'][I] / 2.99e5

    # http://atlas.obs-hp.fr/hyperleda/leda/param/celpos.html
    #I = np.where((0.1*10**miss_hyper['LOGD25'] > 1.) * (miss_hyper['F_ASTROM'] < 1))[0]
    I = np.where((miss_hyper['LOGD25'] > 0.) *
                 (miss_hyper['OBJTYPE'] != 'R') *  # radio source
                 (miss_hyper['OBJTYPE'] != 'PG') * # part of galaxy
                 (miss_hyper['OBJTYPE'] != 'u')    # catalog error
                 )[0]
    log.info(f'Adding {len(I):,d}/{len(miss_hyper):,d} HyperLeda objects with measured diameters ' + \
          'not in ned_hyper and not already in our catalog.')

    miss_hyper = miss_hyper[I]
    basic_miss_hyper = get_basic_geometry(miss_hyper, galaxy_column='OBJNAME_HYPERLEDA', verbose=verbose)

    parent4 = populate_parent(miss_hyper, basic_miss_hyper, verbose=verbose)

    # PGC725719 and WISEA J132257.15-294417.7 (also PGC 725719 in LVD, below)
    # are the same object in NED-LVS and HYPERLEDA, respectively, but their
    # coordinates are 3.5 arcsec apart, so they become two entries; fix that
    # by-hand here.
    log.info('Special-casing PGC725719=WISEA J132257.15-294417.7')
    I = np.where(parent['OBJNAME_NED'] == 'WISEA J132257.15-294417.7')[0]
    J = np.where(parent4['OBJNAME_HYPERLEDA'] == 'PGC725719')[0]
    for col in ['OBJNAME_HYPERLEDA', 'RA_HYPERLEDA', 'DEC_HYPERLEDA', 'Z_HYPERLEDA', 'PGC', 'MAG_HYPERLEDA',
                'BAND_HYPERLEDA', 'BA_HYPERLEDA', 'PA_HYPERLEDA', 'ROW_HYPERLEDA']:
        parent[col][I] = parent4[J][col]
    parent4.remove_row(J[0])

    for col in ['OBJNAME_HYPERLEDA', 'ROW_HYPERLEDA']:
        assert(len(np.unique(parent4[col])) == len(parent4))
    print()
    log.info(f'Parent 4: N={len(parent4):,d}')

    parent = vstack((parent, parent4))

    # [5] Add LVD.
    print()
    log.info('#####')
    log.info(f'Analyzing {len(lvd):,d} LVD objects, of which {len(ned_lvd):,d} ' + \
          f'({100.*len(ned_lvd)/len(lvd):.1f}%) are in ned_lvd.')

    # ned_lvd - already in parent sample
    I = np.where(parent['OBJNAME_NED'] != '')[0]
    #oo, cc = np.unique(parent[I]['OBJNAME_NED'], return_counts=True)
    #p2 = parent2[np.isin(parent2['OBJNAME_NED'], oo[cc>1].value)]
    #p3 = parent3[np.isin(parent3['OBJNAME_NED'], oo[cc>1].value)]
    #p2 = p2[np.argsort(p2['OBJNAME_NED'])]
    #p3 = p3[np.argsort(p3['OBJNAME_NED'])]
    #np.diag(arcsec_between(p2['RA_NED'], p2['DEC_NED'], p3['RA_NED'], p3['DEC_NED']))
    indx_parent, indx_lvd = match(parent[I]['OBJNAME_NED'], ned_lvd['OBJNAME_NED'])

    nexisting = len(indx_parent)
    if verbose:
        log.info(f'Populating ROW_LVD')
    parent['ROW_LVD'][I[indx_parent]] = ned_lvd['ROW'][indx_lvd]
    log.info(f'Matched {len(indx_lvd):,d}/{len(lvd):,d} ({100.*len(indx_lvd)/len(lvd):.1f}%) ' + \
          'ned_lvd objects to the /existing/ parent sample using OBJNAME_NED.')

    indx_parent2, indx_lvd2 = match(parent['ROW_LVD'][I[indx_parent]], lvd['ROW'])
    if verbose:
        for col in ['OBJNAME_LVD', 'RA_LVD', 'DEC_LVD', 'PGC']:
            log.info(f'Populating {col}')
    parent['OBJNAME_LVD'][I[indx_parent[indx_parent2]]] = lvd['OBJNAME'][indx_lvd2]
    parent['RA_LVD'][I[indx_parent[indx_parent2]]] = lvd['RA'][indx_lvd2]
    parent['DEC_LVD'][I[indx_parent[indx_parent2]]] = lvd['DEC'][indx_lvd2]
    parent['PGC'][I[indx_parent[indx_parent2]]] = lvd['PGC'][indx_lvd2]
    #parent[I[indx_parent[indx_parent2]]]['OBJNAME_NED', 'OBJNAME_LVD', 'RA_NED', 'DEC_NED', 'RA_LVD', 'DEC_LVD', 'ROW_LVD', 'PGC']
    #join(parent[I[indx_parent[indx_parent2]]]['OBJNAME_LVD', 'OBJNAME_HYPERLEDA', 'ROW_LVD', 'PGC'], lvd[indx_lvd2]['OBJNAME', 'PGC'], keys_left='OBJNAME_LVD', keys_right='OBJNAME').plog.info(max_lines=-1)

    # NB: These 9 objects are not in my 'hyper' sample because they fail the
    # f_astrom cut; but they're all in the LVD and NED-LVS samples

    #  OBJNAME_LVD  OBJNAME_HYPERLEDA ROW_LVD  PGC_1     OBJNAME     PGC_2
    #     Antlia II                         0 6775392     Antlia II 6775392
    #    Bootes III                         5 4713562    Bootes III 4713562
    #      Cetus II                        14 6740632      Cetus II 6740632
    #     Crater II                        17 5742923     Crater II 5742923
    #       Grus II                        24 6740630       Grus II 6740630
    # Reticulum III                        44 6740628 Reticulum III 6740628
    #   Sagittarius                        45 4689212   Sagittarius 4689212
    #     Tucana IV                        55 6740629     Tucana IV 6740629
    #      Tucana V                        56 6740631      Tucana V 6740631

    # next, match on PGC to pick up the HyperLeda (non-NED) matches
    # 'KK 177': 'IC 4107', # HyperLeda matches to WISEA J130241.76+215952.2
    # 'Leo I 09': 'NGC 3368:[CVD2018] DF6', # HyperLeda matches to SDSS J104653.19+124441.4
    # 'CenA-MM-Dw3': 'Centaurus A:[CSS2016] MM-Dw03', # HyperLeda matches to WISEA J133020.71-421130.6
    # 'JKB142': 'WISEA J014548.01+162239.4', # NED resolves JKB142, but HyperLeda incorrectly matches to WISEA J014548.01+162239.4 by position

    I = np.where((parent['ROW_LVD'] == -99) * (parent['PGC'] != -99))[0]
    J = np.where(lvd['PGC'] > 0)[0]
    indx_parent, indx_lvd = match(parent[I]['PGC'], lvd[J]['PGC'])
    #parent[I[indx_parent]]['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_LVD', 'RA_NED', 'DEC_NED', 'RA_HYPERLEDA', 'DEC_HYPERLEDA', 'RA_LVD', 'DEC_LVD', 'ROW_LVD', 'PGC']
    #lvd[J[indx_lvd]]['OBJNAME', 'RA', 'DEC', 'PGC']
    log.info(f'Matched an additional {len(indx_lvd):,d} LVD objects to the parent sample using PGC.')

    if verbose:
        for col in ['OBJNAME_LVD', 'RA_LVD', 'DEC_LVD', 'ROW_LVD']:
            log.info(f'Populating {col}')
    parent['OBJNAME_LVD'][I[indx_parent]] = lvd['OBJNAME'][J[indx_lvd]]
    parent['RA_LVD'][I[indx_parent]] = lvd['RA'][J[indx_lvd]]
    parent['DEC_LVD'][I[indx_parent]] = lvd['DEC'][J[indx_lvd]]
    parent['ROW_LVD'][I[indx_parent]] = lvd['ROW'][J[indx_lvd]]

    # Drop SDSS J014548.23+162240.6=WISEA J014548.01+162239.4=JKB142. This
    # object appears twice in the NED-LVS (rows 1827699 and 1871170), although
    # NED resolves it as two objects. If we don't drop it then we end up with
    # duplicate LVD entries.
    log.info('Removing duplicate SDSS J014548.23+162240.6=WISEA J014548.01+162239.4=JKB142 from the parent sample.')
    Idrop = np.where(parent['OBJNAME_NED'] == 'SDSS J014548.23+162240.6')[0]
    parent.remove_row(Idrop[0])

    # Also drop GALEXASC J095848.78+665057.9, which appears to be a
    # NED-LVS shred of the LVD dwarf d0958+66=KUG 0945+670 on rows
    # 75946 and 1822204, respectively. If we don't drop it here then
    # SGA2020 incorrectly matches to GALEXASC J095848.78+665057.9.
    log.info('Removing GALEXASC J095848.78+665057.9 from the parent sample.')
    Idrop = np.where(parent['OBJNAME_NED'] == 'GALEXASC J095848.78+665057.9')[0]
    parent.remove_row(Idrop[0])

    # Additional NED drops:
    #   SDSS J002041.45+083701.2 - duplicate with HIPASS J0021+08=JKB129
    #   SDSS J104653.19+124441.4=PGC4689210 - duplicate with Leo dw A=Leo I 09
    #   PGC1 5067061 NED001 - duplicate with Andromeda XXXIII=ANDROMEDA33=Perseus I
    #   SDSS J095549.64+691957.4 - duplicate with SDSSJ141708.23+134105.7=PGC2801015=JKB83
    #   SDSS J133230.32+250724.9 - duplicate with AGC 238890
    #   SDSS J104701.35+125737.5 - duplicate with PGC1 0032256 NED034=LeG21
    #   SDSS J124354.70+412724.9 - duplicate with SMDG J1243552+412727=LV J1243+4127

    dups = ['SDSS J002041.45+083701.2', 'SDSS J104653.19+124441.4', 'PGC1 5067061 NED001',
            'SDSS J095549.64+691957.4', 'SDSS J133230.32+250724.9',
            'SDSS J104701.35+125737.5', 'SDSS J124354.70+412724.9']
    log.info(f'Removing {", ".join(dups)} from the OBJNAME_NED parent sample.')
    Idrop = np.where(np.isin(parent['OBJNAME_NED'], dups))[0]
    parent.remove_rows(Idrop)

    #dups = ['Andromeda XXXIII', 'PGC1 5067061 NED001', 'HIPASS J0021+08', 'SDSS J002041.45+083701.2', 'SDSS J095549.64+691957.4', 'SDSS J141708.23+134105.7', 'SDSS J104653.19+124441.4', 'Leo dw A']
    #bb = parent[np.isin(parent['OBJNAME_NED'], dups)]['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_LVD', 'RA_NED', 'RA_HYPERLEDA', 'RA_LVD', 'DEC_NED', 'DEC_HYPERLEDA', 'DEC_LVD', 'PGC']
    #bb = bb[np.argsort(bb['PGC'])]

    # HyperLeda drops: [CVD2018]M96-DF10 matches SMDG J1048359+130336
    # in NED but not dw1048p1303 in LVD. And NGC3628DGSAT1 matches
    # SMDG J1121369+132650 in NED but not dw1121p1326.
    drops = ['[CVD2018]M96-DF10', 'NGC3628DGSAT1']
    log.info(f'Removing {", ".join(drops)} from the OBJNAME_HYPERLEDA parent sample.')
    Idrop = np.where(np.isin(parent['OBJNAME_HYPERLEDA'], drops))[0]
    parent.remove_rows(Idrop)

    # ned_lvd - not in parent sample (new)
    miss_lvd = ned_lvd[~np.isin(ned_lvd['ROW'], parent['ROW_LVD'])]
    miss_lvd.rename_columns(['OBJNAME', 'ROW'], ['OBJNAME_LVD', 'ROW_LVD'])
    miss_lvd.rename_columns(['RA', 'DEC', 'Z'], ['RA_NED', 'DEC_NED', 'Z_NED'])
    log.info(f'Adding {len(miss_lvd):,d}/{len(lvd):,d} ({100.*len(miss_lvd)/len(lvd):.1f}%) ' + \
          '/new/ ned_lvd objects to the parent sample.')

    basic_miss_lvd = get_basic_geometry(miss_lvd, galaxy_column='OBJNAME_LVD', verbose=verbose)
    parent5a = populate_parent(miss_lvd, basic_miss_lvd, verbose=verbose)

    # LVD - not in parent sample (new)
    miss_lvd = lvd[np.logical_and(~np.isin(lvd['ROW'], parent['ROW_LVD']),
                                  ~np.isin(lvd['ROW'], parent5a['ROW_LVD']))]
    miss_lvd.rename_columns(['OBJNAME', 'ROW'], ['OBJNAME_LVD', 'ROW_LVD'])
    miss_lvd.rename_columns(['RA', 'DEC'], ['RA_LVD', 'DEC_LVD'])
    log.info(f'Adding {len(miss_lvd):,d}/{len(lvd):,d} ({100.*len(miss_lvd)/len(lvd):.1f}%) ' + \
          'new LVD objects to the parent sample.')

    basic_miss_lvd = get_basic_geometry(miss_lvd, galaxy_column='OBJNAME_LVD', verbose=verbose)

    parent5b = populate_parent(miss_lvd, basic_miss_lvd, verbose=verbose)

    parent5 = vstack((parent5a, parent5b))

    # Fill in a bit more info.
    indx_parent, indx_lvd = match(parent5['ROW_LVD'], lvd['ROW'])
    log.info(f'Matching {len(indx_parent):,d} objects to the original LVD catalog.')
    if verbose:
        for col in ['OBJNAME_LVD (replacing the NED-friendly names)', 'RA_LVD', 'DEC_LVD']:
            log.info(f'Populating {col}')
    parent5['OBJNAME_LVD'][indx_parent] = lvd['OBJNAME'][indx_lvd] # replace the NED-friendly names
    parent5['RA_LVD'][indx_parent] = lvd['RA'][indx_lvd]
    parent5['DEC_LVD'][indx_parent] = lvd['DEC'][indx_lvd]

    print()
    log.info(f'Parent 5: N={len(parent5)+nexisting:,d}')

    # [6] Add SGA2020
    parent = vstack((parent, parent5))

    print()
    log.info('#####')
    log.info(f'Analyzing {len(sga2020):,d} SGA-2020 objects.')

    # cross-ID error in HyperLeda which affected the SGA-2020. IC 3881
    # (PGC 43961) in HyperLeda is actually IC 3887, whereas IC 3881 is
    # actually 2MASXJ12545391+1907050 (PGC 3798359) in HyperLeda. Hack
    # the PGC number otherwise we get a grossly overestimated
    # diameter.
    sga2020['PGC'][sga2020['GALAXY'] == 'IC3881'] = 3798359
    sga2020['PGC'][sga2020['GALAXY'] == '2MASXJ12545391+1907050'] = -1

    # match by PGC; any missing objects are due to a variety of errors
    # in the SGA2020 / HyperLeda (I think...)
    I = np.where(parent['PGC'] > 0)[0]
    J = np.where(sga2020['PGC'] > 0)[0]
    log.info(f'Trimmed to {len(J):,d}/{len(sga2020):,d} ({100.*len(J)/len(sga2020):.1f}%) ' + \
          'SGA-2020 objects with PGC>0.')
    sga2020 = sga2020[J]

    #pgc, cc = np.unique(parent[I]['PGC'].value, return_counts=True)
    #dups = pgc[cc>1]
    #parent[np.isin(parent['PGC'], dups)]['OBJNAME_NED', 'OBJNAME_LVD', 'PGC', 'RA_NEDLVS', 'RA_LVD', 'RA_HYPERLEDA', 'DEC_NEDLVS', 'DEC_LVD', 'DEC_HYPERLEDA', 'ROW_NEDLVS', 'ROW_LVD', 'ROW_HYPERLEDA']

    indx_parent, indx_sga2020 = match(parent[I]['PGC'], sga2020['PGC'])
    log.info(f'Matched {len(indx_sga2020):,d}/{len(sga2020):,d} ({100.*len(indx_sga2020)/len(sga2020):.1f}%) ' + \
          'SGA-2020 objects using PGC.')

    for col, sgacol in zip(['OBJNAME_SGA2020', 'RA_SGA2020', 'DEC_SGA2020', 'DIAM_SGA2020', 'BA_SGA2020', 'PA_SGA2020', 'ROW_SGA2020'],
                           ['GALAXY', 'RA', 'DEC', 'D26', 'BA', 'PA', 'ROW']):
        if verbose:
            log.info(f'Populating {sgacol} --> {col}')
        parent[col][I[indx_parent]] = sga2020[indx_sga2020][sgacol]

    I = np.where(parent['ROW_SGA2020'] > 0)[0]
    miss_sga2020 = sga2020[~np.isin(sga2020['ROW'], parent[I]['ROW_SGA2020'])]
    #miss_sga2020.write('junk-sga2020.fits', overwrite=True)
    for col in ['OBJNAME_SGA2020', 'ROW_SGA2020']:
        try:
            assert(len(np.unique(parent[I][col])) == len(parent[I]))
        except:
            log.info(f'Duplicate {col} values!')
            pdb.set_trace()

    # [7] include the custom-added objects plus the SMDG sample
    print()
    log.info('#####')
    log.info(f'Adding {len(custom):,d} more objects from the custom catalog.')

    custom.rename_column('ROW', 'ROW_CUSTOM')

    # update the smudges diameters by a factor of 1.2 to go from Re
    # --> RHolm; this should have been in parse-smudges...
    prefix = np.array(list(zip(*np.char.split(custom['OBJNAME_NED'].value, ' ').tolist()))[0])
    custom['DIAM'][prefix == 'SMDG'] *= 1.2

    # pre-emptively drop three NED-LVS objects which would otherwise
    # be duplicate matches to the SMDG catalog
    drop =['[GMG2015] G', 'SDSS J122357.73+115332.3', 'SMDG J1408411+565536']
    log.info('Dropping '+'.'.join(drop)+' to avoid duplicate matches to the SMUDGes catalog')
    parent.remove_rows(np.where(np.isin(parent['OBJNAME_NED'], drop))[0])

    log.info('Removing GALEXASC J092844.50+360101.8 from the parent sample.')
    Idrop = np.where(parent['OBJNAME_NEDLVS'] == 'GALEXASC J092844.50+360101.8')[0]
    parent.remove_row(Idrop[0])

    ra, dec = parent['RA_NED'].value, parent['DEC_NED'].value
    I = (ra == -99.) * (parent['RA_HYPERLEDA'] != -99.)
    if np.any(I):
        ra[I] = parent['RA_HYPERLEDA'][I].value
        dec[I] = parent['DEC_HYPERLEDA'][I].value
    I = (ra == -99.) * (parent['RA_LVD'] != -99.)
    if np.any(I):
        ra[I] = parent['RA_LVD'][I].value
        dec[I] = parent['DEC_LVD'][I].value

    prefix = np.array(list(zip(*np.char.split(custom['OBJNAME_NED'].value, ' ').tolist()))[0])
    smdg = custom[prefix == 'SMDG']

    m1, m2, sep = match_radec(ra, dec, smdg['RA'], smdg['DEC'], 5./3600., nearest=True)
    log.info(f'Matched {len(m1):,d}/{len(smdg):,d} SMDGes objects to the existing ' + \
          'parent catalog within 5 arcsec.')
    #parent[m1]['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_LVD', 'DIAM_LIT', 'DIAM_LIT_REF', 'ROW_LVD'].plog.info(max_lines=-1)

    basic_smdg = get_basic_geometry(smdg[m2], galaxy_column='OBJNAME_NED', verbose=verbose)
    basic_smdg.rename_column('GALAXY', 'OBJNAME_NED')
    basic_smdg['MAG_LIT_REF'] = 'SMUDGes'
    basic_smdg['DIAM_LIT_REF'] = 'SMUDGes'
    basic_smdg['BA_LIT_REF'] = 'SMUDGes'
    basic_smdg['PA_LIT_REF'] = 'SMUDGes'
    basic_smdg['ROW_CUSTOM'] = smdg[m2]['ROW_CUSTOM']
    #parent6 = populate_parent(smdg[m2], basic_smdg, verbose=verbose)
    for col in basic_smdg.colnames:
        if verbose:
            log.info(f'Populating {col}')
        parent[col][m1] = basic_smdg[col]

    # Now add the rest of the 'custom' sample, making sure to change
    # the 'reference' columns for the remaining SMUDGes sources.
    custom = custom[~np.isin(custom['ROW_CUSTOM'], basic_smdg['ROW_CUSTOM'])]
    basic_custom = get_basic_geometry(custom, galaxy_column='OBJNAME_NED', verbose=verbose)
    parent6 = populate_parent(custom, basic_custom, verbose=verbose)

    prefix = np.array(list(zip(*np.char.split(custom['OBJNAME_NED'].value, ' ').tolist()))[0])    
    parent6['MAG_LIT_REF'][prefix == 'SMDG'] = 'SMUDGes'
    parent6['DIAM_LIT_REF'][prefix == 'SMDG'] = 'SMUDGes'
    parent6['BA_LIT_REF'][prefix == 'SMDG'] = 'SMUDGes'
    parent6['PA_LIT_REF'][prefix == 'SMDG'] = 'SMUDGes'

    print()
    log.info(f'Parent 6: N={len(parent6):,d}')

    # [8] build the final sample
    parent = vstack((parent, parent6))

    # sort, check for uniqueness, and then write out
    srt = np.lexsort((parent['ROW_HYPERLEDA'].value, parent['ROW_NEDLVS'].value,
                      parent['ROW_SGA2020'].value, parent['ROW_LVD'].value, 
                      parent['ROW_CUSTOM'].value))
    parent = parent[srt]

    for col in ['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_NEDLVS', 'OBJNAME_SGA2020', 'OBJNAME_LVD']:
        I = parent[col] != ''
        try:
            assert(len(parent[I]) == len(np.unique(parent[col][I])))
        except:
            log.info(f'Problem with column {col}!')
            pdb.set_trace()
            obj, cc = np.unique(parent[col][I], return_counts=True)

    I = parent['PGC'] > 0
    assert(len(parent[I]) == len(np.unique(parent['PGC'][I])))

    #pgc, count = np.unique(parent['PGC'][I], return_counts=True)
    #bb = parent[np.isin(parent['PGC'], pgc[count>1].value)]['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_NEDLVS',
    #                                                        'OBJNAME_LVD', 'RA_NED', 'DEC_NED', 'PGC', 'ROW_HYPERLEDA', 'ROW_NEDLVS', 'ROW_LVD']
    #bb = bb[np.argsort(bb['PGC'])]

    for col in ['ROW_HYPERLEDA', 'ROW_NEDLVS', 'ROW_SGA2020', 'ROW_LVD', 'ROW_CUSTOM']:
        I = parent[col] != -99
        try:
            assert(len(parent[I]) == len(np.unique(parent[col][I])))
        except:
            log.info(f'Duplicate entries of {col}!')
            pdb.set_trace()

    print()
    log.info('#####')
    log.info(f'Final parent sample: N={len(parent):,d}')
    #parent.write('/global/cfs/cdirs/desicollab/users/ioannis/SGA/2025/parent/external/junk-sga2020.fits', overwrite=True)

    # Populate OBJNAME, RA, DEC, and Z. Prefer LVD coordinates and use SGA2020
    # before HyperLeda, otherwise we totally miss some quite famous galaxies.
    print()

    for dataset in ['LVD', 'NED', 'NEDLVS', 'SGA2020', 'HYPERLEDA']:
        I = np.where((parent['RA'] == -99.) * (parent[f'RA_{dataset}'] != -99.))[0]
        if len(I) > 0:
            log.info(f'Adopting {len(I):,d}/{len(parent):,d} ({100.*len(I)/len(parent):.1f}%) ' + \
                  f'RA,DEC values from {dataset}.')
            parent['RA'][I] = parent[I][f'RA_{dataset}']
            parent['DEC'][I] = parent[I][f'DEC_{dataset}']


    # NB - prefer LVD then NED names
    for dataset in ['LVD', 'NED', 'NEDLVS', 'SGA2020', 'HYPERLEDA']:
        I = np.where((parent['OBJNAME'] == '') * (parent[f'OBJNAME_{dataset}'] != ''))[0]
        if len(I) > 0:
            log.info(f'Adopting {len(I):,d}/{len(parent):,d} ({100.*len(I)/len(parent):.1f}%) ' + \
                  f'OBJNAMEs from {dataset}.')
            # Roughly 4300 objects have "SDSSJ" names rather than "SDSS J".
            # Standardize that here so we match NED-LVS and so that we're
            # NED-friendly.
            if dataset == 'HYPERLEDA':
                objname = nedfriendly_hyperleda(parent[I][f'OBJNAME_{dataset}'])
                # put a space after the prefix, to better match NED
                prefix = np.array([re.findall(r'\d*\D+', obj)[0] for obj in objname])
                newobjname = objname.copy()
                for iobj, (obj, pre) in enumerate(zip(objname, prefix)):
                    if not 'SDSS' in pre:
                        newobjname[iobj] = obj.replace(pre, f'{pre} ')
                parent['OBJNAME'][I] = newobjname
            else:
                parent['OBJNAME'][I] = parent[I][f'OBJNAME_{dataset}']

    # NB - prefer NEDLVS redshifts
    I = np.isnan(parent['Z_HYPERLEDA'])
    if np.any(I):
        parent['Z_HYPERLEDA'][I] = -99.
    for dataset in ['NEDLVS', 'NED', 'HYPERLEDA']:
        I = np.where((parent['Z'] == -99.) * (parent[f'Z_{dataset}'] != -99.))[0]
        if len(I) > 0:
            log.info(f'Adopting {len(I):,d}/{len(parent):,d} ({100.*len(I)/len(parent):.1f}%) Z values from {dataset}.')
            parent['Z'][I] = parent[I][f'Z_{dataset}']
    print()

    # reset the values from the literature and then prioritize
    basic_ned_hyper = get_basic_geometry(ned_hyper, galaxy_column='ROW', verbose=verbose)
    basic_ned_nedlvs = get_basic_geometry(ned_nedlvs, galaxy_column='ROW', verbose=verbose)
    basic_lvd = get_basic_geometry(lvd, galaxy_column='ROW', verbose=verbose)
    basic_custom = get_basic_geometry(custom, galaxy_column='ROW_CUSTOM', verbose=verbose)

    # NB: do not reset diameters for the set of SMUDGes objects which
    # were matched from custom to the parent sample (at the point
    # where 'custom' was added to parent, parent 6).
    I = parent['DIAM_LIT_REF'] != 'SMUDGes'
    for col in ['DIAM_LIT', 'BA_LIT', 'PA_LIT', 'MAG_LIT']:
        parent[col][I] = -99.
        parent[f'{col}_REF'][I] = ''
    parent['BAND_LIT'][I] = ''

    print()
    # NB - prioritize LVD first, then custom (which includes SMUDGes)
    for basic, row, dataset in zip((basic_lvd, basic_custom, basic_ned_hyper, basic_ned_nedlvs),
                                   ('ROW_LVD', 'ROW_CUSTOM', 'ROW_HYPERLEDA', 'ROW_NEDLVS'),
                                   ('LVD', 'CUSTOM', 'NED-HyperLeda', 'NEDLVS')):
        for col in ['DIAM_LIT', 'BA_LIT', 'PA_LIT', 'MAG_LIT']:
            I = np.where((parent[col] == -99.) * (parent[row] != -99))[0]
            if len(I) > 0:
                # 'GALAXY' here is actually 'ROW'
                indx_parent, indx_basic = match(parent[I][row], basic['GALAXY'])
                G = np.where(basic[indx_basic][col] != -99.)[0]
                if len(G) > 0:
                    log.info(f'Populating parent with {len(G):,d}/{len(I):,d} {col}s from {dataset}.')
                    parent[col][I[indx_parent[G]]] = basic[indx_basic[G]][col]
                    parent[f'{col}_REF'][I[indx_parent[G]]] = basic[indx_basic[G]][f'{col}_REF']
                    if col == 'MAG_LIT':
                        parent['BAND_LIT'][I[indx_parent[G]]] = basic[indx_basic[G]]['BAND_LIT']
                    #parent[I[indx_parent[G]]]['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_NEDLVS', 'OBJNAME_LVD', 'DIAM_LIT', 'DIAM_LIT_REF', 'BA_LIT', 'BA_LIT_REF', 'PA_LIT', 'PA_LIT_REF']

        print()

    # special columns for HyperLeda and SGA2020 geometry
    basic_hyper = get_basic_geometry(hyper, galaxy_column='ROW', verbose=verbose)
    basic_sga2020 = get_basic_geometry(sga2020, galaxy_column='ROW', verbose=verbose)

    for basic, row, suffix in zip((basic_hyper, basic_sga2020), 
                                  ('ROW_HYPERLEDA', 'ROW_SGA2020'), 
                                  ('HYPERLEDA', 'SGA2020')):
        for col in [f'DIAM_{suffix}', f'BA_{suffix}', f'PA_{suffix}', f'MAG_{suffix}']:
            parent[col] = -99.
        parent[f'BAND_{suffix}'] = ''

        I = np.where(parent[row] != -99)[0]
        if len(I) > 0:
            # 'GALAXY' here is actually 'ROW'
            indx_parent, indx_basic = match(parent[I][row], basic['GALAXY']) 
            for col in [f'DIAM_{suffix}', f'BA_{suffix}', f'PA_{suffix}', f'MAG_{suffix}']:
                log.info(f'Populating parent with {len(I):,d} {col}s from {suffix}.')
                parent[col][I[indx_parent]] = basic[indx_basic][col]

    # final statistics
    nobj = len(parent)
    for prop in ['DIAM', 'BA', 'PA', 'MAG']:
        col = f'{prop}_LIT'
        N = parent[col] != -99.
        refs = np.unique(parent[N][f'{col}_REF'])
        log.info(f'N({col}) = {np.sum(N):,d}/{nobj:,d} ({100.*np.sum(N)/nobj:.1f}%)')
        for ref in refs:
            R = parent[N][f'{col}_REF'] == ref
            log.info(f'  N({ref}) = {np.sum(R):,d}/{np.sum(N):,d} ({100.*np.sum(R)/np.sum(N):.1f}%)')

        for ref in ['HYPERLEDA', 'SGA2020']:
            col = f'{prop}_{ref}'
            N = parent[col] != -99.
            log.info(f'N({col}) = {np.sum(N):,d}/{nobj:,d} ({100.*np.sum(N)/nobj:.1f}%)')
        print()

    parent['ROW_PARENT'] = np.arange(len(parent))

    version = parent_version(nocuts=True)
    outfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-nocuts-{version}.fits')
    log.info(f'Writing {len(parent):,d} objects to {outfile}')
    parent.meta['EXTNAME'] = 'PARENT-NOCUTS'
    parent.write(outfile, overwrite=True)
