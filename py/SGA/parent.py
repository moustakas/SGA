"""
SGA.parent
==========

Code for defining the SGA parent sample.

"""
import os, time, sys, re, pdb
import numpy as np
import fitsio
from glob import glob
from importlib import resources
from collections import Counter, namedtuple
from astropy.table import Table, vstack, hstack
from astrometry.libkd.spherematch import match_radec

from SGA.SGA import sga_dir, SGA_version
from SGA.coadds import PIXSCALE, BANDS
from SGA.util import match, match_to
from SGA.sky import choose_primary, resolve_close
from SGA.qa import qa_skypatch, multipage_skypatch

from SGA.logger import log


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
    parent['OBJNAME_DR910'] = np.zeros(nobj, '<U30')

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
    parent['RA_DR910'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_DR910'] = np.zeros(nobj, 'f8') -99.

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
    parent['ROW_DR910'] = np.zeros(nobj, '<i8') -99

    return parent


def check_lvd(cat, lvdcat=None):
    from SGA.external import read_lvd
    lvd = read_lvd(verbose=False)
    if lvdcat is None:
        lvdcat = cat[cat['ROW_LVD'] != -99]
    lvdmiss = None
    try:
        assert(len(lvd[~np.isin(lvd['ROW'], lvdcat['ROW_LVD'])]) == 0)
    except:
        log.info('Missing the following LVD systems!')
        lvdmiss = lvd[~np.isin(lvd['ROW'], lvdcat['ROW_LVD'])]
    return lvdmiss


def qa_parent(nocuts=False, sky=False, size_mag=False):
    """QA of the parent sample.

    """
    from SGA.qa import fig_sky, fig_size_mag

    qadir = os.path.join(sga_dir(), 'parent', 'qa')
    if not os.path.isdir(qadir):
        os.makedirs(qadir)

    if nocuts:
        version = SGA_version(nocuts=True)
        suffix = '-nocuts'
    else:
        version = SGA_version(parent=True)
        suffix = ''
    catfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent{suffix}-{version}.fits')

    cat = Table(fitsio.read(catfile))#, rows=np.arange(10000)))
    log.info(f'Read {len(cat):,d} objects from {catfile}')

    if sky:
        png = os.path.join(qadir, f'qa-sky-parent{suffix}-{version}.png')
        #I = cat['DIAM_LIT'] > 0.2
        fig_sky(cat, racolumn='RA', deccolumn='DEC', pngfile=png,
                clip_lo=0., clip_hi=300., mloc=50.)

    if size_mag:
        png = os.path.join(qadir, f'qa-sizemag-parent{suffix}-{version}.png')
        fig_size_mag(cat, nocuts=nocuts, pngfile=png)

    # compare diameters
    if False:
        pngfile = os.path.join(qadir, 'qa-diamleda.png')
        I = np.where((allcat['DIAM_LIT'] > 0.) * (allcat['DIAM_HYPERLEDA'] > mindiam))[0]
        log.info(len(I))
        lim = (-1.7, 3.)
        #lim = (np.log10(mindiam)-0.1, 3.)

        import corner
        from SGA.qa import plot_style
        sns, colors = plot_style(talk=True, font_scale=1.2)

        fig, ax = plt.subplots(figsize=(7, 7))
        xx = np.log10(allcat['DIAM_HYPERLEDA'][I])
        yy = np.log10(allcat['DIAM_LIT'][I])
        J = ['IRAS' in morph for morph in allcat['MORPH'][I]]
        corner.hist2d(xx, yy, levels=[0.5, 0.75, 0.95, 0.995],
                      bins=100, smooth=True, color=colors[0], ax=ax, # mpl.cm.get_cmap('viridis'),
                      plot_density=True, fill_contours=True, range=(lim, lim),
                      data_kwargs={'color': colors[0], 'alpha': 0.4, 'ms': 4},
                      contour_kwargs={'colors': 'k'},)
        ax.scatter(xx[J], yy[J], s=15, marker='x', color=colors[2])
        #ax.scatter(xx, yy, s=10)
        ax.set_xlabel(r'$\log_{10}$ (Diameter) [HyperLeda]')
        ax.set_ylabel(r'$\log_{10}$ (Diameter) [archive]')
        #ax.set_xlim(lim)
        #ax.set_ylim(lim)
        ax.plot(lim, lim, color='k', lw=2)
        fig.tight_layout()
        log.info(f'Writing {pngfile}')
        fig.savefig(pngfile)#, bbox_inches='tight')
        plt.close(fig)

        I = np.where(allcat['DIAM_LIT'] > mindiam)[0]
        log.info(f'Trimming to {len(I):,d}/{len(allcat):,d} ({100.*len(I)/len(allcat):.1f}%) ' + \
              f'objects with DIAM_LIT>{60.*mindiam:.1f} arcsec.')
        cat = allcat[I]

        srt = np.argsort(cat['DIAM_LIT'])[::-1]
        cat[srt]['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_LVD', 'DIAM_LIT', 'DIAM_LIT_REF', 'MORPH']


def qa_footprint(region='dr9-north', show_fullcat=False, show_fullccds=False):
    """Build some simple QA of the in-footprint sample.

    TODO - make this a sky map

    """
    from collections import Counter
    import matplotlib.pyplot as plt
    from SGA.qa import plot_style

    sns, colors = plot_style(talk=True, font_scale=0.9)

    version = SGA_version(archive=True)

    catfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-archive-{region}-{version}.fits')
    cat = Table(fitsio.read(catfile, columns=['OBJNAME', 'RA', 'DEC', 'ROW_PARENT', 'FILTERS']))
    log.info(f'Read {len(cat):,d} objects from {catfile}')

    qafile = os.path.join(sga_dir(), 'parent', 'qa', f'qa-parent-archive-{region}-{version}.png')

    fig, ax = plt.subplots(figsize=(8, 6))

    if show_fullcat:
        fullcatfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-archive-{version}.fits')
        fullcat = Table(fitsio.read(fullcatfile, columns=['OBJNAME', 'RA', 'DEC', 'ROW_PARENT']))
        log.info(f'Read {len(fullcat):,d} objects from {fullcatfile}')
        ax.scatter(fullcat['RA'], fullcat['DEC'], s=1, color='gray')

    if show_fullccds:
        from legacypipe.runs import get_survey
        from SGA.io import set_legacysurvey_dir
        from SGA.coadds import RUNS

        set_legacysurvey_dir(region)
        survey = get_survey(RUNS[region], allbands=BANDS[region])
        _ = survey.get_ccds_readonly()
        allccds = survey.ccds
        log.info(f'Read {len(allccds):,d} CCDs from region={region}')
        I = allccds.ccd_cuts == 0
        log.info(f'Trimming to {np.sum(I):,d}/{len(allccds):,d} CCDs with ccd_cuts==0')
        allccds = allccds[I]

        ax.scatter(allccds.ra, allccds.dec, s=1, color='gray')

    #for bands in sorted(set(cat['FILTERS'])):
    C = Counter(cat['FILTERS'])
    for iband, bands in enumerate(sorted(C, reverse=True)):
        I = cat['FILTERS'] == bands
        if np.sum(I) > 0:
            ax.scatter(cat['RA'][I], cat['DEC'][I], s=1, alpha=0.7, zorder=iband,
                       label=f'{bands} (N={np.sum(I):,d})')
    ax.set_xlabel('R.A. (degree)')
    ax.set_ylabel('Dec (degree)')
    if False: # M31 + family
        ax.set_xlim(15., 8.)
        ax.set_ylim(30., 50.)
    else:
        ax.set_xlim(360., 0.)
        if region == 'dr9-north':
            ax.set_ylim(-25., 90.)
            loc = 'lower left'
        else:
            ax.set_ylim(-95., 95.)
            loc = 'upper left'
    ax.legend(fontsize=9, ncols=4, markerscale=8, loc=loc)
    fig.tight_layout()
    fig.savefig(qafile)
    log.info(f'Wrote {qafile}')


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

    cols = ['OBJNAME', 'OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'MORPH', 'DIAM_LIT', 'DIAM_HYPERLEDA',
            'OBJTYPE', 'RA', 'DEC', 'RA_NED', 'DEC_NED', 'RA_HYPERLEDA', 'DEC_HYPERLEDA',
            'MAG_LIT', 'Z', 'PGC', 'ROW_PARENT']

    crossidfile = resources.files('SGA').joinpath('data/SGA2025/SGA2025-crossid-errors.csv')

    # Read or, optionally, rebuild the cross-id error file.
    if rebuild_file:
        VETO = [
            'NGC 3280', # will get dropped
            #'2MASS J04374625-2711389', # 2MASS J04374625-2711389 and WISEA J043745.83-271135.1 are distinct.
        ]

        # First, resolve 1-arcsec pairs **excluding** GPair and GTrpl
        # systems and LVD sources.
        I = (fullcat['OBJTYPE'] != 'GPair') * (fullcat['OBJTYPE'] != 'GTrpl') * (fullcat['ROW_LVD'] == -99)
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


def update_lvd_properties(cat):
    """Gather geometry for all LVD sources.

       name            ra            dec      rhalf
      str22         float64        float64   float64
    ---------- ------------------ ---------- -------
     dw1341-29 205.33416666666665   -29.5675      --
         IC239             39.116     38.969      --
      NGC 1042          2.6733275  -8.433591      --
      NGC 4151         12.1757335 39.4057938      --
      NGC 4424         12.4532467  9.4204423      --
      NGC 5194         202.469625  47.195167      --
    PGC 100170          2.9477205 58.9115793      --
    PGC 166192         20.5090597 60.3540088      --
    PGC 166193         20.5255533 60.8123555      --
      UGC 7490          186.10375  70.334278      --

    """
    def modify_lvg_names(lvg_name):
        lvg_name = np.char.replace(lvg_name, ' ', '')
        lvg_name = np.char.replace(lvg_name, 'SagdSph', 'Sagittarius')
        lvg_name = np.char.replace(lvg_name, 'And ', 'Andromeda ')
        lvg_name = np.char.replace(lvg_name, 'Lac', 'Lacerta')
        lvg_name = np.char.replace(lvg_name, 'UMa', 'UrsaMajor')
        lvg_name = np.char.replace(lvg_name, 'UMin', 'UrsaMinor')
        lvg_name = np.char.replace(lvg_name, 'Hydrus1dw', 'HydrusI')
        lvg_name = np.char.replace(lvg_name, 'UGCA086', 'UGCA86')
        lvg_name = np.char.replace(lvg_name, 'Antlia2', 'AntliaII')
        lvg_name = np.char.replace(lvg_name, 'Horologium2', 'HorologiumII')
        lvg_name = np.char.replace(lvg_name, 'ColumbiaI', 'ColumbaI')
        #lvg_name = np.char.replace(lvg_name, 'Pegasus', 'PegasusdIrr')
        lvg_name[lvg_name == 'Pegasus'] = 'PegasusdIrr'
        return lvg_name


    from astropy.table import vstack, join, hstack
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astrometry.libkd.spherematch import match_radec
    from SGA.util import match
    from SGA.external import read_lvd, version_lvd, nedfriendly_lvd

    _lvd = read_lvd()
    version = version_lvd()
    nobj = len(_lvd)

    if version != 'v1.0.5':
        log.warning('Be sure to review hard-coded properties before proceeding!')

    log.info(f'Updating properties of {nobj} LVD galaxies.')

    #indx_cat = np.where(cat['ROW_LVD'] != -99)[0]
    #lvd = cat[indx_cat]

    lvd = Table()
    lvd['OBJNAME'] = _lvd['OBJNAME']
    lvd['RA'] = _lvd['RA']
    lvd['DEC'] = _lvd['DEC']
    lvd['ROW'] = _lvd['ROW']
    lvd['DIAM'] = np.zeros(nobj, 'f4') - 99.
    lvd['PA'] = np.zeros(nobj, 'f4') - 99.
    lvd['BA'] = np.zeros(nobj, 'f4') - 99.
    lvd['RADEC_REF'] = np.zeros(nobj, 'U19')
    lvd['DIAM_REF'] = np.zeros(nobj, 'U19')
    lvd['PA_REF'] = np.zeros(nobj, 'U19')
    lvd['BA_REF'] = np.zeros(nobj, 'U19')

    lvd['RADEC_REF'] = 'LVD'

    I = np.isfinite(_lvd['RHALF'])
    lvd['DIAM'][I] = np.float32(2. * 1.2 * _lvd['RHALF'][I])
    lvd['DIAM_REF'][I] = version

    I = np.isfinite(_lvd['POSITION_ANGLE'])
    lvd['PA'][I] = np.float32(_lvd['POSITION_ANGLE'][I] % 180)
    lvd['PA_REF'][I] = version

    I = np.isfinite(_lvd['ELLIPTICITY'])
    lvd['BA'][I] = np.float32(1. - _lvd['ELLIPTICITY'][I])
    lvd['BA_REF'][I] = version

    # first, add geometry from the LVGDB; see bin/analyze-lvd
    lvgfile = resources.files('SGA').joinpath('data/SGA2025/lvg_table1_2025-01-26_trim.txt')
    lvg = Table.read(lvgfile, format='ascii.fixed_width')
    log.info(f'Read the properties of {len(lvg)} galaxies from {lvgfile}')

    c_lvg = SkyCoord(ra=lvg['RA_HMS'].value, dec=lvg['DEC_DMS'].value, unit=(u.hourangle, u.deg))
    lvg['RA'] = c_lvg.ra.degree
    lvg['DEC'] = c_lvg.dec.degree
    lvg.remove_columns(['RA_HMS', 'DEC_DMS'])

    [lvg.rename_column(col, f'{col}_LVG') for col in lvg.colnames]

    m1, m2, _ = match_radec(lvd['RA'], lvd['DEC'], lvg['RA_LVG'], lvg['DEC_LVG'], 30./3600., nearest=True)
    log.info(f'Matched {len(m1)}/{len(lvd)} LVD objects based on coordinates.')
    #hstack((lvd['OBJNAME', 'RA', 'DEC'][m1], lvg['NAME_LVG', 'RA_LVG', 'DEC_LVG'][m2])).plog.info(max_lines=-1)

    lvd['DIAM_LVG'] = np.zeros(nobj, 'f4') - 99.
    lvd['BA_LVG'] = np.zeros(nobj, 'f4') - 99.
    #lvd['pa_LVG'] = np.zeros(nobj, 'f4')

    lvd['DIAM_LVG'][m1] = lvg['DIAM_LVG'][m2]
    lvd['BA_LVG'][m1] = lvg['BA_LVG'][m2]
    #lvd['PA_LVG'][m1] = lvg['PA_LVG'][m2]

    # now try matching missing matches by name
    I = np.where(lvd['DIAM_LVG'] == -99.)[0]
    lvg_name = modify_lvg_names(lvg['NAME_LVG'].value)
    m1, m2 = match(np.char.replace(lvd['OBJNAME'][I], ' ', ''), lvg_name)
    log.info(f'Matched {len(m1)}/{len(lvd)} LVD objects based on object name.')
    #hstack((lvd['OBJNAME', 'RA', 'DEC'][I][m1], lvg['NAME_LVG', 'RA_LVG', 'DEC_LVG'][m2])).plog.info(max_lines=-1)

    lvd['DIAM_LVG'][I[m1]] = lvg['DIAM_LVG'][m2]
    lvd['BA_LVG'][I[m1]] = lvg['BA_LVG'][m2]
    #lvd['pa_LVG'][m1] = lvg['pa_LVG'][m2]

    # copy missing values from the LVGDB (prefer LVGDB!)
    #log.info(f'Existing diameters for {np.sum(lvd["DIAM"] == -99.)}/{len(lvd)} objects.')
    #I = (lvd['DIAM'] == -99.) * (lvd['DIAM_LVG'] != -99.)
    I = (lvd['DIAM_LVG'] != -99.)
    log.info(f'Adding {np.sum(I)} diameters from the LVGDB.')
    lvd['DIAM'][I] = lvd['DIAM_LVG'][I]
    lvd['BA'][I] = lvd['BA_LVG'][I]
    lvd['DIAM_REF'][I] = 'LVGDB'
    lvd['BA_REF'][I] = 'LVGDB'

    # Next, gather ellipticities and position angles from the input catalog.
    I = np.where(cat['ROW_LVD'] != -99)[0]
    m1, m2 = match(lvd['ROW'], cat['ROW_LVD'][I])
    lvd = lvd[m1]
    lvdcat = cat[I[m2]]

    for prop in ['BA', 'PA']:
        for ref in ['SGA2020', 'HYPERLEDA', 'LIT']:
            I = (lvd[prop] == -99.) * (lvdcat[f'{prop}_{ref}'] != -99.)
            if np.any(I):
                log.info(f'Adding {np.sum(I)} {prop}s from {ref}')
                lvd[prop][I] = lvdcat[f'{prop}_{ref}'][I]
                if ref == 'LIT':
                    lvd[f'{prop}_REF'][I] = lvdcat[f'{prop}_{ref}_REF'][I]
                else:
                    lvd[f'{prop}_REF'][I] = ref

    # default values for everything else
    I = lvd['PA'] == -99.
    if np.any(I):
        log.info(f'Adopting default PAs for {np.sum(I)} objects.')
        lvd['PA'][I] = 0.
        lvd['PA_REF'][I] = 'default'

    I = lvd['BA'] == -99.
    if np.any(I):
        log.info(f'Adopting default BAs for {np.sum(I)} objects.')
        lvd['BA'][I] = 1.
        lvd['BA_REF'][I] = 'default'

    # override the algorithm above based on VI
    if version == 'v1.0.5':
        updatefile = resources.files('SGA').joinpath(f'data/SGA2025/LVD-geometry-updates-{version}.csv')
        updates = Table.read(updatefile, format='csv', comment='#')
        log.info(f'Read {len(updates)} objects from {updatefile}')

        I = match_to(lvd['OBJNAME'].value, updates['objname'].value)
        assert(np.all(lvd['OBJNAME'][I] == updates['objname'].value))

        log.info('Updating hard-coded properties:')
        for col in ['RA', 'DEC', 'DIAM', 'BA', 'PA']:
            J = np.where(updates[col.lower()] != -99.)[0]
            if len(J) > 0:
                log.info(f'  --{len(J)} {col} values')
                lvd[col][I[J]] = updates[col.lower()][J]
                if col == 'RA' or col == 'DEC':
                    lvd['RADEC_REF'][I[J]] = updates['radec_ref'][J]
                else:
                    lvd[f'{col}_REF'][I[J]] = updates[f'{col.lower()}_ref'][J]

    assert(np.all((lvd['DIAM'] != -99.) * (lvd['BA'] != -99.)))
    #lvd[lvd['PA_REF'] == 'default']

    # one last update -- set b/a=1 if pa_ref=='default'
    I = (lvd['PA_REF'] == 'default') * (lvd['BA'] != 1.)
    if np.any(I):
        lvd['BA'][I] = 1.

    # write out
    lvd = lvd[np.argsort(lvd['ROW'])]

    outfile = os.path.join(sga_dir(), 'parent', 'external', f'LVD_{version}_geometry.fits')
    lvd.write(outfile, overwrite=True)
    log.info(f'Wrote {len(lvd)} objects to {outfile}')

    #if not os.path.isfile(outfile):
    #    log.info(f'Wrote {len(lvd)} objects to {outfile}')
    #    lvd.write(outfile, overwrite=True)
    #else:
    #    log.info(f'Output file {outfile} exists.')

    # Finally, populate the catalog.
    I = np.where(cat['ROW_LVD'] != -99)[0]
    m1, m2 = match(lvd['ROW'], cat['ROW_LVD'][I])
    for prop in ['RA', 'DEC']:
        cat[f'{prop}'][I[m2]] = lvd[prop][m1]
    for prop in ['DIAM', 'BA', 'PA']:
        cat[f'{prop}_LIT'][I[m2]] = lvd[prop][m1]
        cat[f'{prop}_LIT_REF'][I[m2]] = 'LVD' # lvd[f'{prop}_REF'][m1]

    return cat


def update_properties(cat, verbose=False):
    """Update properties, including incorrect coordinates, in both NED and
    HyperLeda.

    """
    # First, update the LVD objects.
    # [0] Update LVD properties.
    cat = update_lvd_properties(cat)

    propfile = resources.files('SGA').joinpath('data/SGA2025/SGA2025-vi-properties.csv')
    props = Table.read(propfile, format='csv', comment='#')

    out = cat.copy()

    allnewcols = ['ra', 'dec', 'diam', 'pa', 'ba', 'diam_hyperleda']
    allupdatecols = ['RA', 'DEC', 'DIAM_LIT', 'PA_LIT', 'BA_LIT', 'DIAM_HYPERLEDA']

    for prop in props:
        objname = prop['objname_ned']
        I = np.where(objname == cat['OBJNAME'].value)[0]
        if len(I) != 1:
            log.info(f'Problem finding {objname}!')
            pdb.set_trace()
            raise ValueError(f'Problem finding {objname}!')
        if verbose:
            log.info(f'{objname} ({cat["OBJTYPE"][I[0]]}):')

        if prop['update_hyperleda_coords'] == 'Yes':
            newcols = allnewcols + ['ra', 'dec']
            updatecols = allupdatecols + ['RA_HYPERLEDA', 'DEC_HYPERLEDA']
        else:
            newcols = allnewcols
            updatecols = allupdatecols

        for newcol, col in zip(newcols, updatecols):
            newval = prop[newcol]
            oldval = cat[col][I[0]]
            if verbose:
                if newval != -99.:
                    log.info(f'  Updating {col}: {oldval} --> {newval}')
                else:
                    log.info(f'  Retaining {col}: {oldval}')
            if newval != -99.:
                out[col][I] = newval

    # some SGA2020 diameters are grossly over/underestimated; fix those here
    objs = ['NGC 0134', 'NGC 4157', 'NGC 3254', 'NGC 4312', 'NGC 4666',
            'NGC 4178', 'ESO 358- G 063', 'NGC 3254', 'NGC 4257', 'NGC 2549',
            'NGC 4010', 'NGC 5899', 'NGC 7541', 'NGC 4062',
            'MESSIER 085', 'UGC 00484', 'NGC 0720', 'ESO 079- G 003', 'NGC 7721',
            'NGC 3923', 'NGC 6902', 'NGC 4266', 'ESO 186-IG 069 NED02', 'NGC 5859',
            'NGC 4395', 'UGC 12732', 'MCG -02-55-005', 'NGC 4383',
            'ESO 114- G 008', 'WISEA J022103.95-334348.4', 'UGC 10046 NED02', 'UGC 10046 NED01',
            'ESO 234- G 024', 'UGC 09621', 'UGC 10988', 'NGC 6654A',
            'NGC 2460', 'UGC 08524', 'UGC 09883', 'NGC 0765', 'ARK 018',
            'VCC 0377', 'UGC 01382', 'UGC 08011', 'ESO 245- G 006',
            'IC 0774', 'NGC 4701', 'ESO 119- G 048', 'ESO 108- G 023',
            'NGC 5248', 'ESO 306- G 009', 'NGC 1403', 'NGC 1437A',
            'NGC 4797', 'MCG -01-35-020', 'NGC 0692', 'NGC 6902B',
            'UGC 07178', 'NGC 4816', ]
    diams = [16., 12., 6., 6., 7., # [arcmin
             7., 7., 6., 7., 6.,
             5.5, 5.5, 6., 6.,
             12., 5., 9.5, 4.5, 5.5,
             10., 9., 5., 3., 4.,
             5., 5., 2., 3.5,
             2., 0.3, 1.5, 1.5,
             2., 1., 1.5, 4.,
             5., 2., 1., 4.5, 2.,
             2., 3.7, 2.2, 3.,
             2.2, 3.2, 3.4, 3.5,
             9., 2.8, 4., 3.,
             2.6, 4., 4., 2.5,
             2.5, 3., ]
    for obj, diam in zip(objs, diams):
        I = cat['OBJNAME'] == obj
        if np.sum(I) == 1:
            old = out['DIAM_SGA2020'][I][0]
            log.info(f'Updating the SGA2020 diameter: {obj}: {old:.3f}-->{diam:.3f} arcmin')
            out['DIAM_SGA2020'][I] = diam

    return out


def in_footprint_work(allcat, chunkindx, allccds, comm=None, radius=1.,
                      width_pixels=38, bands=BANDS):
    """Support routine for in_footprint.

    """
    from SGA.sky import get_ccds

    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    if rank == 0:
        from SGA.mpi import weighted_partition

        t0 = time.time()

        # work on a "chunk" of the catalog
        cat = allcat[chunkindx]

        # I, which is a list of len(cat), is the variable-length of indices into
        # allccds of the matches, or None if no match (which we filter out).
        allindx_ccds = match_radec(cat['RA'].value, cat['DEC'].value, allccds.ra,
                                   allccds.dec, radius, indexlist=True)

        indx_cat = []
        indx_ccds = []
        nccdperobj = []
        for icat, indx_ccds1 in enumerate(allindx_ccds):
            if indx_ccds1 is not None:
                indx_cat.append(icat)
                indx_ccds.append(indx_ccds1)
                nccdperobj.append(len(indx_ccds1))
        nccdperobj = np.array(nccdperobj)
        indx_cat = np.array(indx_cat)

        nobj = len(indx_cat)
        if nobj > 0:
            log.info(f'  Found {len(indx_cat):,d}/{len(cat):,d} objects with at ' + \
                  f'least one CCD within {radius} degree')

            work_byrank = weighted_partition(nccdperobj, size)
            log.info(f'  Distributing {len(np.unique(np.hstack(indx_cat))):,d} objects and ' + \
                  f'{len(np.unique(np.hstack(indx_ccds))):,d} CCDs to {size:,d} ranks')
        else:
            log.info(f'  No objects with at least one CCD within {radius} degree')
    else:
        nobj = 0

    if comm:
        nobj = comm.bcast(nobj, root=0)

    if nobj == 0:
        return Table(), Table()

    if comm:
        # Rank 0 sends work...
        if rank == 0:
            for onerank in range(1, size):
                #log.info(f'Rank {rank} sending work to rank {onerank}')
                if len(work_byrank[onerank]) > 0:
                    comm.send(cat[indx_cat[work_byrank[onerank]]], dest=onerank, tag=1)
                    # build and then send the per-object tables of CCDs
                    ccds_onerank = []
                    for work_byrank_onegal in work_byrank[onerank]:
                        ccds_onerank.append(allccds[indx_ccds[work_byrank_onegal]])
                    comm.send(ccds_onerank, dest=onerank, tag=2)
                else:
                    comm.send(Table(), dest=onerank, tag=1)
                    comm.send(Table(), dest=onerank, tag=2)
            # work for rank 0
            cat_onerank = cat[indx_cat[work_byrank[rank]]]
            ccds_onerank = []
            for work_byrank_onegal in work_byrank[rank]:
                ccds_onerank.append(allccds[indx_ccds[work_byrank_onegal]])
            #log.info(f'Rank {rank} received {len(cat_onerank):,d} objects.')
        else:
            # ...and the other ranks receive the work.
            cat_onerank = comm.recv(source=0, tag=1)
            ccds_onerank = comm.recv(source=0, tag=2)
            #log.info(f'Rank {rank} received {len(cat_onerank):,d} objects.')
    else:
        if len(work_byrank[rank]) > 0:
            cat_onerank = cat[indx_cat[work_byrank[rank]]]
            ccds_onerank = []
            for work_byrank_onegal in work_byrank[rank]:
                ccds_onerank.append(allccds[indx_ccds[work_byrank_onegal]])
        else:
            cat_onerank = Table()
            ccds_onerank = Table()

    if comm:
        comm.barrier()

    # now perform a more refined per-object search for matching CCDs
    fcat = []
    fccds = []
    nobj = len(cat_onerank)
    #log.info(rank, nobj)

    #if nobj == 0:
    #    log.info(f'Rank {rank}: all done; no work to do.')
    if nobj > 0:
        t1 = time.time()
        #log.info(f'Rank {rank}: gathering CCDs for {nobj:,d} objects.')
        for icat, (cat_onegal, ccds_onegal) in enumerate(zip(cat_onerank, ccds_onerank)):
            #if icat % 1000 == 0:
            #    log.info(f'Rank {rank}: Working on galaxy: {icat:,d}/{nobj:,d}')
            fccds1, fcat1 = get_ccds(ccds_onegal, cat_onegal, width_pixels,
                                     pixscale=PIXSCALE, return_ccds=True)
            fcat.append(fcat1)
            fccds.append(fccds1)
        if len(fcat) > 0:
            fcat = vstack(fcat)
            fccds = vstack(fccds)
        else:
            fcat = Table()
            fccds = Table()
        #log.info(f'Rank {rank}: all done in {(time.time()-t1)/60.:.3f} minutes')

    if comm:
        fcat = comm.gather(fcat, root=0)
        fccds = comm.gather(fccds, root=0)

    # sort and return
    if rank == 0:
        fcat = vstack(fcat)
        fccds = vstack(fccds)

        if len(fcat) > 0:
            fcat = fcat[np.argsort(fcat['ROW_PARENT'])]
            fccds = fccds[np.argsort(fccds['ROW_PARENT'])]
            log.info(f'  Gathered {len(fccds):,d} CCDs for {len(fcat):,d}/{len(indx_cat):,d} ' + \
                  f'objects in {time.time()-t0:.2f} sec')

        return fcat, fccds
    else:
        return Table(), Table()


def _read_existing_footprint(cat, region, version='v1.0'):
    """Read existing / previous catalogs of running the footprint code on an
    input catalog.

    """
    catfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-archive-{region}-{version}.fits')
    #ccdsfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-ccds-{region}-{version}.fits')
    if os.path.isfile(catfile):# and os.path.isfile(ccdsfile):
        refcat = Table(fitsio.read(catfile, columns=['OBJNAME', 'RA', 'DEC', 'ROW_PARENT', 'NCCD', 'FILTERS']))
        #refcat = refcat[:1000]

        log.info(f'Read {len(refcat):,d} objects from the existing {region} ' + \
              f'in-footprint catalog {catfile}')

        # We know that objects with sep==0. are in the footprint, so
        # we don't have to "find" them again...
        m1, m2, sep = match_radec(cat['RA'].value, cat['DEC'].value, refcat['RA'].value,
                                  refcat['DEC'].value, 1./3600., nearest=True)
        I = sep == 0.
        if np.any(I):
            m1 = m1[I]
            m2 = m2[I]
            log.info(f'Matched {len(m1):,d}/{len(cat):,d} objects to the existing ' + \
                  f'{region} in-footprint catalog.')
            todo = np.delete(np.arange(len(cat)), m1)
            cat_todo = cat[todo] # could be an empty set

            refcat_done = refcat[m2]
            cat_done = cat[m1]
            cat_done['NCCD'] = refcat_done['NCCD']
            cat_done['FILTERS'] = refcat_done['FILTERS']

            #ccds_rows = fitsio.read(ccdsfile, columns='ROW_PARENT')
            #rows = np.where(np.isin(ccds_rows, cat_done['ROW_PARENT'].value))[0]
            #ccds_done = Table(fitsio.read(ccdsfile, rows=rows))

            ##ccds_done = ccds_done[:1000]
            #log.info('Matching CCDs...but have to loop!')
            #for ii in range(len(cat_done)):
            #    ccds_done['ROW_PARENT'][refcat_done[ii]['ROW_PARENT'] == ccds_done['ROW_PARENT']] = cat_done[ii]['ROW_PARENT']
            return cat_todo, cat_done#, ccds_done

    return cat, Table()#, Table()


def in_footprint(region='dr9-north', comm=None, radius=1., width_pixels=38,#152,
                 bands=BANDS, ntest=None):
    """Find which objects are in the given survey footprint based on positional
    matching with a very generous (1 deg) search radius.

    radius in degrees
    width_pixels - diameter for more refined search

    """
    from SGA.coadds import RUNS

    if comm is None:
        rank = 0
    else:
        rank = comm.rank

    if rank == 0:
        from legacypipe.runs import get_survey
        from SGA.io import set_legacysurvey_dir

        t0 = time.time()

        # read the parent catalog
        version = SGA_version(archive=True)
        catfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-archive-{version}.fits')

        F = fitsio.FITS(catfile)
        N = F[1].get_nrows()
        if ntest is not None:
            rng = np.random.default_rng(seed=1)
            I = rng.choice(N, size=ntest, replace=False)
            I = I[np.argsort(I)]
        else:
            #log.info('HACK TO JUST SELECT THE RC3!!')
            #ref = fitsio.read(catfile, columns='DIAM_LIT_REF')
            #I = np.where(ref == 'RC3')[0]
            I = np.arange(N)

        #log.info('HACK!!')
        #I = np.array([4784167, 4784170])
        #I = np.array([2016776])

        cat = Table(fitsio.read(catfile, columns=['OBJNAME', 'RA', 'DEC', 'ROW_PARENT'], rows=I))
        cat['NCCD'] = np.zeros(len(cat), int)
        cat['FILTERS'] = np.zeros(len(cat), '<U4')
        log.info(f'Read {len(cat):,d} objects from {catfile}')

        # sort by right ascension to try to speed things up...
        cat = cat[np.argsort(cat['RA'])]

        set_legacysurvey_dir(region)
        survey = get_survey(RUNS[region], allbands=BANDS[region])

        _ = survey.get_ccds_readonly()
        allccds = survey.ccds
        log.info(f'Read {len(allccds):,d} CCDs from region={region}')
        I = allccds.ccd_cuts == 0
        log.info(f'Trimming to {np.sum(I):,d}/{len(allccds):,d} CCDs with ccd_cuts==0')
        allccds = allccds[I]

        # in dr9-north, censor the CCDs in and around M31 and one failed brick around (ra,dec)=(328.8, 0.0)
        # https://www.legacysurvey.org/viewer-desi?ra=10.6890&dec=41.2719&layer=unwise-neo7&zoom=9&ccds9n
        # https://www.legacysurvey.org/viewer-desi?ra=328.8798&dec=0.0422&layer=ls-dr9-north&zoom=12&ccds9n&bricks
        # 3288p000
        if region == 'dr9-north':
            I = (allccds.ra > 8.5) * (allccds.ra < 12.5) * (allccds.dec > 35.) * (allccds.dec < 45.)
            log.info(f'Removing {np.sum(I)} CCDs around M31.')
            allccds = allccds[~I]

            B = survey.get_brick_by_name('3288p000')
            I = (cat['RA'] > B.ra1) * (cat['RA'] < B.ra2) * (cat['DEC'] > B.dec1) * (cat['DEC'] < B.dec2)
            if np.count_nonzero(I) > 0:
                log.info(f'Removing {np.sum(I):,d} objects located in failed brick 3288p000')
                cat = cat[~I]

        sys.stdout.flush()

        # check for an existing CCDs catalog from previous runs of this code
        #cat, cat_done, ccds_done = _read_existing_footprint(cat, allccds, region, version=version)
        cat, cat_done = _read_existing_footprint(cat, region, version=version)
        #log.info('HACK!!')
        #cat_done = []

        # divide the parent catalog into more managable chunks, and loop
        if region == 'dr9-north':
            nperchunk = 100000
        else:
            nperchunk = 10000
        nobj = len(cat)
        if nobj == 0: # if already done
            nchunk = 1
        else:
            nchunk = int(np.ceil(nobj / nperchunk))
        chunkindx = np.array_split(np.arange(nobj), nchunk)
    else:
        cat = None
        chunkindx = None
        allccds = None

    if comm:
        chunkindx = comm.bcast(chunkindx, root=0)

    allfcat = []
    #allfccds = []
    for ichunk, indx in enumerate(chunkindx):
        if rank == 0:
            log.info(f'Working on chunk {ichunk+1:,d}/{nchunk:,d} with {len(indx):,d} objects')
        fcat, fccds = in_footprint_work(cat, indx, allccds, comm=comm, radius=radius,
                                        width_pixels=width_pixels, bands=bands)
        sys.stdout.flush()

        if rank == 0:
            allfcat.append(fcat)
            #allfccds.append(fccds)

    if comm:
        comm.barrier()

    # gather up the results and write out
    if rank == 0:
        from SGA.util import match
        allfcat = vstack(allfcat)
        #allfccds = vstack(allfccds)

        # is there an existing catalog?
        if len(cat_done) > 0:
            allfcat = vstack((allfcat, cat_done))
            #allfccds = vstack((allfccds, ccds_done))

        outcat = Table(fitsio.read(catfile)) # read the whole catalog
        # match on position not ROW_PARENT, which can change
        cat_indx, fcat_indx, sep = match_radec(
            outcat['RA'].value, outcat['DEC'].value, allfcat['RA'].value,
            allfcat['DEC'].value, 1./3600., nearest=True)
        #cat_indx, fcat_indx = match(outcat['ROW_PARENT'], allfcat['ROW_PARENT'])
        outcat = outcat[cat_indx]
        allfcat = allfcat[fcat_indx]
        assert(np.all(outcat['ROW_PARENT'] == allfcat['ROW_PARENT']))

        outcat['NCCD'] = allfcat['NCCD']
        outcat['FILTERS'] = allfcat['FILTERS']
        outcat = outcat[np.argsort(outcat['ROW_PARENT'])]

        version = SGA_version(archive=True)
        outfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-archive-{region}-{version}.fits')
        log.info(f'Writing {len(outcat):,d} objects to {outfile}')
        outcat.write(outfile, overwrite=True)

        ## need to figure this out...
        #if 'col0' in allfccds.columns:
        #    allfccds.remove_column('col0')

        #allfccds = allfccds[np.argsort(allfccds['ROW_PARENT'])]
        #ccdsfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-ccds-{region}-{version}.fits')
        #log.info(f'Writing {len(allfccds):,d} CCDs to {ccdsfile}')
        #allfccds.write(ccdsfile, overwrite=True)

        log.info(f'All done in {(time.time()-t0)/60.:.2f} min')


def build_parent_nocuts(verbose=True, overwrite=False):
    """Merge the external catalogs from SGA2025-query-ned.

    """
    import re
    from astropy.table import join
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    import astropy.units as u

    from SGA.geometry import get_basic_geometry
    from SGA.external import (read_hyperleda, version_hyperleda, nedfriendly_hyperleda,
                              read_hyperleda_galaxies, version_hyperleda_galaxies,
                              read_hyperleda_multiples, version_hyperleda_multiples,
                              read_hyperleda_noobjtype, version_hyperleda_noobjtype,
                              read_nedlvs, version_nedlvs, read_sga2020, read_lvd,
                              read_dr910, read_custom_external, version_lvd, nedfriendly_lvd)


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

    version_nocuts = SGA_version(nocuts=True)
    final_outfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-nocuts-{version_nocuts}.fits')
    if os.path.isfile(final_outfile) and not overwrite:
        log.info(f'Parent catalog {final_outfile} exists; use --overwrite')
        return


    log.info('#####')
    log.info('Input data:')
    dr910 = read_dr910()
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

    #miss_lvd = lvd[~np.isin(lvd['ROW'], ned_lvd['ROW'])

    #####
    #I = np.where(lvd['PGC'] > 0)[0]
    #pgc, cc = np.unique(lvd['PGC'][I].value, return_counts=True)
    #check = lvd[I][cc>1]
    #check = check[np.argsort(check['PGC'])]
    ####

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

    #oldpgc = parent['PGC'][I[indx_parent[indx_parent2]]].value
    #newpgc = lvd['PGC'][indx_lvd2].value
    #_check = (oldpgc != -99) * (newpgc != 0) * (newpgc != -1)
    #check = hstack((parent[I[indx_parent[indx_parent2]]]['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'RA_NED', 'DEC_NED', 'RA_HYPERLEDA', 'DEC_HYPERLEDA', 'PGC'][_check], lvd[indx_lvd2]['OBJNAME', 'OBJNAME_NED', 'RA', 'DEC', 'PGC'][_check]))

    #parent[I[indx_parent[indx_parent2]]]['OBJNAME_NED', 'OBJNAME_LVD', 'RA_NED', 'DEC_NED', 'RA_LVD', 'DEC_LVD', 'ROW_LVD', 'PGC']
    #join(parent[I[indx_parent[indx_parent2]]]['OBJNAME_LVD', 'OBJNAME_HYPERLEDA', 'ROW_LVD', 'PGC'], lvd[indx_lvd2]['OBJNAME', 'PGC'], keys_left='OBJNAME_LVD', keys_right='OBJNAME').plog.info(max_lines=-1)

    ####
    #I = np.where(parent['PGC'] > 0)[0]
    #pgc, cc = np.unique(parent['PGC'][I].value, return_counts=True)
    #if np.any(cc>1):
    #    log.warning('Duplicate PGC values!!')
    #    check = parent[I][np.isin(parent['PGC'][I], pgc[cc>1])]
    #    check = check[np.argsort(check['PGC'])]
    #    check = check['OBJNAME_NED', 'OBJNAME_LVD', 'PGC', 'RA_NEDLVS', 'RA_LVD', 'RA_HYPERLEDA', 'DEC_NEDLVS', 'DEC_LVD', 'DEC_HYPERLEDA', 'ROW_NEDLVS', 'ROW_LVD', 'ROW_HYPERLEDA']
    #
    #    pdb.set_trace()
    #
    #K = np.isin(parent['PGC'][I], dups)
    #check = parent['OBJNAME_NED', 'OBJNAME_LVD', 'PGC', 'RA_NEDLVS', 'RA_LVD', 'RA_HYPERLEDA', 'DEC_NEDLVS', 'DEC_LVD', 'DEC_HYPERLEDA', 'ROW_NEDLVS', 'ROW_LVD', 'ROW_HYPERLEDA'][I][K]
    #check = check[np.argsort(check['PGC'])]
    #pdb.set_trace()
    ####

    # NB: These 9 objects are not in my 'hyper' sample because they fail the
    # f_astrom cut; but they're all in the LVD and NED-LVS samples

    #  OBJNAME_LVD  OBJNAME_HYPERLEDA ROW_LVD  PGC_1     OBJNAME     PGC_2
    #     Antlia II                         0 6775392     Antlia II 6775392
    #    Bootes III                         5 4713562    Bootes III 4713562
    #      Cetus II                        14 6740632      Cetus II 6740632
    #     Crater II                        17 5742923     Crater II 5742923
    #       Grus II                        24 6740630       Grus II 6740630
    # Reticulum III                        44 6740628 Reticulum III 6740628
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

    # Drop duplicates / incorrect matches.
    #I = np.where(parent['PGC'] > 0)[0]
    #pgc, cc = np.unique(parent['PGC'][I].value, return_counts=True)
    #dups = pgc[cc>1]
    #K = np.isin(parent['PGC'][I], dups)
    #check = parent['OBJNAME_NED', 'OBJNAME_LVD', 'PGC', 'RA_NEDLVS', 'RA_LVD', 'RA_HYPERLEDA', 'DEC_NEDLVS', 'DEC_LVD', 'DEC_HYPERLEDA', 'ROW_NEDLVS', 'ROW_LVD', 'ROW_HYPERLEDA'][I][K]
    #check = check[np.argsort(check['PGC'])]
    #
    #   SDSS J133230.32+250724.9 - duplicate with AGC 238890
    #   SDSS J002041.45+083701.2 - duplicate with HIPASS J0021+08=JKB129
    #   SDSS J104701.35+125737.5 - duplicate with PGC1 0032256 NED034=LeG21
    #   SDSS J104653.19+124441.4=PGC4689210 - duplicate with Leo dw A=Leo I 09
    #   SDSS J124354.70+412724.9 - duplicate with SMDG J1243552+412727=LV J1243+4127
    #   PGC1 5067061 NED001 - duplicate with Andromeda XXXIII=ANDROMEDA33=Perseus I
    #   SDSS J014548.23+162240.6 - duplicate with WISEA J014548.01+162239.4=JKB142
    #   SDSS J095549.64+691957.4 - duplicate with SDSSJ141708.23+134105.7=PGC2801015=JKB83
    dups = ['SDSS J133230.32+250724.9', 'SDSS J002041.45+083701.2',
            'SDSS J104701.35+125737.5', 'SDSS J104653.19+124441.4',
            'SDSS J124354.70+412724.9', 'PGC1 5067061 NED001',
            'SDSS J014548.23+162240.6', 'SDSS J095549.64+691957.4',
            ]
    log.info(f'Removing {", ".join(dups)} from the OBJNAME_NED parent sample.')
    Idrop = np.where(np.isin(parent['OBJNAME_NED'], dups))[0]
    parent.remove_rows(Idrop)

    # Also drop GALEXASC J095848.78+665057.9, which appears to be a
    # NED-LVS shred of the LVD dwarf d0958+66=KUG 0945+670 on rows
    # 75946 and 1822204, respectively. If we don't drop it here then
    # SGA2020 incorrectly matches to GALEXASC J095848.78+665057.9.
    log.info('Removing GALEXASC J095848.78+665057.9 from the parent sample.')
    Idrop = np.where(parent['OBJNAME_NED'] == 'GALEXASC J095848.78+665057.9')[0]
    parent.remove_row(Idrop[0])

    ## Additional NED drops:
    ##   PGC1 0040904 NED002 - duplicate with LV J1243+4127
    ##   SDSS J104654.61+124717.5 - duplicate with LeG21
    #dups = [, 'PGC1 0040904 NED002', 'SDSS J104654.61+124717.5']
    #log.info(f'Removing {", ".join(dups)} from the OBJNAME_NED parent sample.')
    #Idrop = np.where(np.isin(parent['OBJNAME_NED'], dups))[0]
    #parent.remove_rows(Idrop)

    #dups = ['Andromeda XXXIII', 'PGC1 5067061 NED001', 'HIPASS J0021+08', 'SDSS J002041.45+083701.2', 'SDSS J095549.64+691957.4', 'SDSS J141708.23+134105.7', 'SDSS J104653.19+124441.4', 'Leo dw A']
    #bb = parent[np.isin(parent['OBJNAME_NED'], dups)]['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_LVD', 'RA_NED', 'RA_HYPERLEDA', 'RA_LVD', 'DEC_NED', 'DEC_HYPERLEDA', 'DEC_LVD', 'PGC']
    #bb = bb[np.argsort(bb['PGC'])]

    # HyperLeda drops:
    #   [CVD2018]M96-DF10 matches SMDG J1048359+130336 in NED but not dw1048p1303 in LVD
    #   NGC3628DGSAT1 matches SMDG J1121369+132650 in NED but not dw1121p1326
    drops = ['[CVD2018]M96-DF10', 'NGC3628DGSAT1']
    log.info(f'Removing {", ".join(drops)} from the OBJNAME_HYPERLEDA parent sample.')
    Idrop = np.where(np.isin(parent['OBJNAME_HYPERLEDA'], drops))[0]
    parent.remove_rows(Idrop)

    ####
    #I = np.where(parent['PGC'] > 0)[0]
    #pgc, cc = np.unique(parent['PGC'][I].value, return_counts=True)
    #dups = pgc[cc>1]
    #K = np.isin(parent['PGC'][I], dups)
    #check = parent['OBJNAME_NED', 'OBJNAME_LVD', 'PGC', 'RA_NEDLVS', 'RA_LVD', 'RA_HYPERLEDA', 'DEC_NEDLVS', 'DEC_LVD', 'DEC_HYPERLEDA', 'ROW_NEDLVS', 'ROW_LVD', 'ROW_HYPERLEDA'][I][K]
    #check = check[np.argsort(check['PGC'])]
    #pdb.set_trace()
    ####

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

    ####
    #pgc, cc = np.unique(parent['PGC'][I].value, return_counts=True)
    #dups = pgc[cc>1]
    #K = np.isin(parent['PGC'], dups)
    #check = parent['OBJNAME_NED', 'OBJNAME_LVD', 'PGC', 'RA_NEDLVS', 'RA_LVD', 'RA_HYPERLEDA', 'DEC_NEDLVS', 'DEC_LVD', 'DEC_HYPERLEDA', 'ROW_NEDLVS', 'ROW_LVD', 'ROW_HYPERLEDA'][K]
    #check = check[np.argsort(check['PGC'])]
    ####

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

    # [7] include the custom-added objects plus the SMDGes sample
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

    # [8] include the DR9/DR10 supplemental objects
    parent = vstack((parent, parent6))

    print()
    log.info('#####')
    log.info(f'Adding {len(dr910):,d} more objects from the DR9/DR10 supplemental catalog.')

    basic_dr910 = get_basic_geometry(dr910, galaxy_column='DESINAME', verbose=verbose)
    dr910.rename_columns(['ROW', 'DESINAME', 'RA', 'DEC'],
                         ['ROW_DR910', 'OBJNAME_DR910', 'RA_DR910', 'DEC_DR910'])
    parent7 = populate_parent(dr910, basic_dr910, verbose=True)#verbose)

    # [9] build the final sample
    parent = vstack((parent, parent7))

    # sort, check for uniqueness, and then write out
    srt = np.lexsort((parent['ROW_HYPERLEDA'].value, parent['ROW_NEDLVS'].value,
                      parent['ROW_SGA2020'].value, parent['ROW_LVD'].value,
                      parent['ROW_CUSTOM'].value, parent['ROW_DR910']))
    parent = parent[srt]

    for col in ['OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'OBJNAME_NEDLVS', 'OBJNAME_SGA2020', 'OBJNAME_LVD', 'OBJNAME_DR910']:
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

    for col in ['ROW_HYPERLEDA', 'ROW_NEDLVS', 'ROW_SGA2020', 'ROW_LVD', 'ROW_CUSTOM', 'ROW_DR910']:
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

    for dataset in ['LVD', 'NED', 'NEDLVS', 'SGA2020', 'HYPERLEDA', 'DR910']:
        I = np.where((parent['RA'] == -99.) * (parent[f'RA_{dataset}'] != -99.))[0]
        if len(I) > 0:
            log.info(f'Adopting {len(I):,d}/{len(parent):,d} ({100.*len(I)/len(parent):.1f}%) ' + \
                  f'RA,DEC values from {dataset}.')
            parent['RA'][I] = parent[I][f'RA_{dataset}']
            parent['DEC'][I] = parent[I][f'DEC_{dataset}']


    # NB - prefer LVD then NED names
    for dataset in ['LVD', 'NED', 'NEDLVS', 'SGA2020', 'HYPERLEDA', 'DR910']:
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
    basic_dr910 = get_basic_geometry(dr910, galaxy_column='ROW_DR910', verbose=verbose)

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
    for basic, row, dataset in zip((basic_lvd, basic_custom, basic_ned_hyper, basic_ned_nedlvs, basic_dr910),
                                   ('ROW_LVD', 'ROW_CUSTOM', 'ROW_HYPERLEDA', 'ROW_NEDLVS', 'ROW_DR910'),
                                   ('LVD', 'CUSTOM', 'NED-HyperLeda', 'NEDLVS', 'DR910')):
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


    # ROW_PARENT must be unique across versions
    rowfiles = glob(os.path.join(sga_dir(), 'parent', f'SGA2025-parent-nocuts-*-rows.fits'))
    if len(rowfiles) > 0:
        log.warning('FIXME!')
        rows = np.arange(len(parent))
        #pdb.set_trace()
        # adjust ROWS so they're unique
    else:
        rows = np.arange(len(parent))

    parent['ROW_PARENT'] = rows

    rowfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-nocuts-{version_nocuts}-rows.fits')
    rowcat = parent['OBJNAME', 'ROW_PARENT', ]
    log.info(f'Writing {len(rowcat):,d} objects to {rowfile}')
    rowcat.write(rowfile, overwrite=True)

    log.info(f'Writing {len(parent):,d} objects to {final_outfile}')
    parent.meta['EXTNAME'] = 'PARENT-NOCUTS'
    parent.write(final_outfile, overwrite=True)


def build_parent_vicuts(verbose=False, overwrite=False):
    """Build the parent catalog with VI cuts.

    Always make sure we still have all LVD sources (see the VETO array
    in remove_by_prefix)!

    """
    version_vicuts = SGA_version(vicuts=True)
    final_outfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-vicuts-{version_vicuts}.fits')
    if os.path.isfile(final_outfile) and not overwrite:
        log.info(f'Parent catalog {final_outfile} exists; use --overwrite')
        return


    # quick check on duplicates
    actionsfile = resources.files('SGA').joinpath('data/SGA2025/SGA2025-vi-actions.csv')
    actions = Table.read(actionsfile, format='csv', comment='#')
    uobj, cc = np.unique(actions['objname'].value, return_counts=True)
    if np.any(cc > 1):
        log.info(f'The following objects are duplicates in {actionsfile}')
        for obj in uobj[cc > 1]:
            log.info(obj)
        return

    version = SGA_version(nocuts=True)
    catfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-nocuts-{version}.fits')
    origcat = Table(fitsio.read(catfile))
    log.info(f'Read {len(origcat):,d} objects from {catfile}')
    lvdmiss = check_lvd(origcat)
    if lvdmiss is not None:
        log.info(lvdmiss)
        pdb.set_trace()

    # [1] Update individual-galaxy properties, including coordinates.
    cat1 = update_properties(origcat, verbose=verbose)
    lvdmiss = check_lvd(cat1)
    if lvdmiss is not None:
        log.info(lvdmiss)
        pdb.set_trace()

    # [2] Drop systems with uncommon prefixes (after VI).
    cat2 = remove_by_prefix(cat1, merger_type=None, verbose=verbose, build_qa=False)
    lvdmiss = check_lvd(cat2)
    if lvdmiss is not None:
        log.info(lvdmiss)
        pdb.set_trace()
    del cat1

    # [3] Resolve cross-identification errors in NED/HyperLeda.
    cat3 = resolve_crossid_errors(cat2, verbose=False, build_qa=False, rebuild_file=True)
    lvdmiss = check_lvd(cat3)
    if lvdmiss is not None:
        log.info(lvdmiss)
        pdb.set_trace()
    del cat2

    # [4] Resolve close (1 arcsec) pairs.
    cat4 = resolve_close(cat3, cat3, maxsep=1., allow_vetos=True, verbose=False)
    lvdmiss = check_lvd(cat4)
    if lvdmiss is not None:
        log.info(lvdmiss)
        pdb.set_trace()
    del cat3

    # [4] Visually drop GTrpl and GPair systems with and without measured
    # diameters.
    cat = remove_by_prefix(cat4, merger_type='GTrpl', merger_has_diameter=False, verbose=verbose, build_qa=False)
    cat = remove_by_prefix(cat, merger_type='GPair', merger_has_diameter=False, verbose=verbose, build_qa=False)
    del cat4

    # [5] only explicitly drop mergers with diameters via VI and the VI-actions file
    remove_by_prefix(cat, merger_type='GTrpl', merger_has_diameter=True, verbose=verbose, build_qa=False)
    remove_by_prefix(cat, merger_type='GPair', merger_has_diameter=True, verbose=verbose, build_qa=False)

    # write out
    log.info(f'Writing {len(cat):,d} objects to {final_outfile}')
    cat.meta['EXTNAME'] = 'PARENT-VICUTS'
    cat.write(final_outfile, overwrite=True)


def add_gaia_masking(cat):

    log.info(f'Adding Gaia bright-star masking bits.')
    if not 'STARFDIST' in cat.colnames:
        cat['STARFDIST'] = np.zeros(len(cat), 'f4') + 99.
    if not 'STARDIST' in cat.colnames:
        cat['STARDIST'] = np.zeros(len(cat), 'f4') + 99.
    if not 'STARMAG' in cat.colnames:
        cat['STARMAG'] = np.zeros(len(cat), 'f4') + 99.

    gaiafile = os.path.join(sga_dir(), 'gaia', 'gaia-mask-dr3-galb9.fits')
    gaia = Table(fitsio.read(gaiafile, columns=['ra', 'dec', 'radius', 'mask_mag', 'isbright', 'ismedium']))
    log.info(f'Read {len(gaia):,d} Gaia stars from {gaiafile}')
    I = gaia['radius'] > 0.
    log.info(f'Trimmed to {np.sum(I):,d}/{len(gaia):,d} stars with radius>0')
    gaia = gaia[I]

    dmag = 1.
    bright = np.min(np.floor(gaia['mask_mag']))
    faint = np.max(np.ceil(gaia['mask_mag']))
    magbins = np.arange(bright, faint, dmag)

    for mag in magbins:
        # find all Gaia stars in this magnitude bin
        I = np.where((gaia['mask_mag'] >= mag) * (gaia['mask_mag'] < mag+dmag))[0]

        # search within 2 times the largest masking radius
        maxradius = 2. * np.max(gaia['radius'][I]) # [degrees]
        log.info(f'Found {len(I):,d} Gaia stars in magnitude bin {mag:.0f} to ' + \
              f'{mag+dmag:.0f} with max radius {maxradius:.4f} degrees.')

        m1, m2, sep = match_radec(cat['RA'], cat['DEC'], gaia['ra'][I], gaia['dec'][I], maxradius, nearest=True)
        if len(m1) > 0:
            zero = np.where(sep == 0.)[0]
            if len(zero) > 0:
                cat['STARDIST'][m1[zero]] = 0.
                cat['STARFDIST'][m1[zero]] = 0.
                cat['STARMAG'][m1[zero]] = gaia['mask_mag'][I[m2[zero]]]

            # separations can be identically zero
            pos = np.where(sep > 0.)[0]
            if len(pos) > 0:
                # distance to the nearest star (in this mag bin)
                # relative to the mask radius of that star (given its
                # mag), capped at a factor of 2; values <1 mean the
                # object is within the star's masking radius
                fdist = sep[pos] / gaia['radius'][I[m2[pos]]].value
                # only store the smallest value
                J = np.where((fdist < 2.) * (fdist < cat['STARFDIST'][m1[pos]]))[0]
                if len(J) > 0:
                    cat['STARDIST'][m1[pos[J]]] = sep[pos[J]] # [degrees]
                    cat['STARFDIST'][m1[pos[J]]] = fdist[J]
                    cat['STARMAG'][m1[pos[J]]] = gaia['mask_mag'][I[m2[pos[J]]]]


def build_parent_archive(verbose=False, overwrite=False):
    """Build the parent catalog.

    """
    from glob import glob
    from astropy.table import join
    from SGA.external import read_custom_external
    from SGA.geometry import choose_geometry

    version_archive = SGA_version(archive=True)
    final_outfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-archive-{version_archive}.fits')
    if os.path.isfile(final_outfile) and not overwrite:
        log.info(f'Parent catalog {final_outfile} exists; use --overwrite')
        return


    version_vicuts = SGA_version(vicuts=True)
    catfile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-vicuts-{version_vicuts}.fits')
    cat = Table(fitsio.read(catfile))
    log.info(f'Read {len(cat):,d} objects from {catfile}')

    lvdmiss = check_lvd(cat)
    if lvdmiss is not None:
        log.info(lvdmiss)
        pdb.set_trace()

    # read the ssl-legacysurvey results (including the veto file)
    log.info('Applying the ssl results')

    # First read the classification results and throw out the
    # reference sources, which are visually inspected.
    sslfiles = glob(os.path.join(sga_dir(), 'ssl', 'v[1,3]', 'output', 'ssl-parent-chunk???-v[1,3].txt'))
    #sslfiles = glob(os.path.join(sga_dir(), 'ssl', 'v?', 'output', 'ssl-parent-chunk???-v?.txt'))
    ssl = vstack([Table.read(sslfile, format='ascii.commented_header') for sslfile in sslfiles])
    _, I = np.unique(ssl['ROW'], return_index=True)
    ssl = ssl[I]
    ssl = ssl[ssl['REF'] == 0]

    # Now, 'row' may have changed, so we need to read the parent
    # sample(s) to get 'objname'.
    sslfiles = glob(os.path.join(sga_dir(), 'ssl', 'ssl-parent-cat-v[1,3].fits'))
    ssl_parent = vstack([Table(fitsio.read(sslfile)) for sslfile in sslfiles])
    _, uindx = np.unique(ssl_parent['OBJNAME'].value, return_index=True)
    ssl_parent = ssl_parent[uindx]

    indx_parent, indx_ssl = match(ssl_parent['ROW_PARENT'], ssl['ROW'])
    ssl_parent = ssl_parent[indx_parent]
    ssl = ssl[indx_ssl]
    ssl['OBJNAME'] = ssl_parent['OBJNAME']
    ssl['REGION'] = ssl_parent['REGION']

    # investigate objects with common names
    #bb = ssl[ssl['REGION'] == 'dr9-north']
    #bb = bb[np.argsort(bb['RA'])]
    #bb[(prefix != 'SDSS') * (prefix != 'WISEA') * (prefix != '2MASS') * (prefix != 'GALEXMSC') * (prefix != 'GALEXASC')].plog.info(max_lines=-1)

    # when testing, the container has two SGA repos in PYTHONPATH
    try:
        vetodir = str(resources.files('SGA').joinpath('data/SGA2025')._paths[0])
    except:
        vetodir = str(resources.files('SGA').joinpath('data/SGA2025'))
    vetofiles = glob(os.path.join(vetodir, 'ssl-veto-v[1,3].txt'))
    veto = vstack([Table.read(vetofile, format='csv', comment='#') for vetofile in vetofiles])

    # Make sure that the objects we're trying to veto are actually in the ssl
    # files.
    I = np.isin(ssl['OBJNAME'], veto['objname'])
    if np.sum(I) != len(veto):
        log.info('Missing objects in veto files!')
        log.info(veto[~np.isin(veto['objname'], ssl['OBJNAME'])])
        pdb.set_trace()
    ssl = ssl[~I]

    # Add to the veto array any objects with an entry in either the
    # 'properties' or (non-drop) 'actions' file.
    propfile = resources.files('SGA').joinpath('data/SGA2025/SGA2025-vi-properties.csv')
    props = Table.read(propfile, format='csv', comment='#')
    I = np.isin(ssl['OBJNAME'].value, props['objname_ned'].value)
    if np.any(I):
        log.info(f'WARNING: need to add the following {np.sum(I):,d} objects to the appropriate veto file')
        log.info(ssl[I])
        pdb.set_trace()

    actionsfile = resources.files('SGA').joinpath('data/SGA2025/SGA2025-vi-actions.csv')
    actions = Table.read(actionsfile, format='csv', comment='#')
    actions = actions[actions['action'] == 'hyperleda-coords']
    I = np.isin(ssl['OBJNAME'].value, actions['objname'].value)
    if np.any(I):
        log.info(f'WARNING: need to add the following {np.sum(I):,d} objects to the appropriate veto file')
        log.info(ssl[I])
        pdb.set_trace()

    log.info(f'Removing {len(ssl):,d}/{len(cat):,d} objects based on SSL results.')
    cat = cat[~np.isin(cat['OBJNAME'], ssl['OBJNAME'])]

    lvdmiss = check_lvd(cat)
    if lvdmiss is not None:
        log.info(lvdmiss)
        pdb.set_trace()

    # Resolve more close pairs -- these choices were made after
    # investigating a bunch of QA.

    # Create refcat by sorting cat by PGC, otherwise RA will be used
    # to define the "first" group member, which is not usually what we
    # want. E.g., UGC 08168 is dropped in favor of 2MASS
    # J13034084+5129425 even though the latter has a larger PGC number
    # (but has sep==0.).
    maxsep = 5.
    log.info('Resolving additional close pairs.')
    refcat =  cat.copy()
    refcat[refcat['PGC'] < 0]['PGC'] = np.max(cat['PGC'].value)+1
    srt = np.argsort(refcat['PGC'])
    refcat = refcat[srt]
    cat = cat[srt]
    #bb = refcat[np.isin(refcat['OBJNAME'], ['UGC 08168', '2MASS J13034084+5129425'])]
    #bb = refcat[np.isin(refcat['OBJNAME'], ['GALEXASC J072041.35+561217.0', 'WISEA J072041.29+561218.0'])]
    cat2 = resolve_close(cat, refcat, maxsep=maxsep, allow_vetos=False,
                         ignore_objtype=True, trim=True, verbose=False)
    lvdmiss = check_lvd(cat2)
    if lvdmiss is not None:
        log.info(lvdmiss)
        pdb.set_trace()
    cat = cat2
    cat = cat[np.argsort(cat['RA'].value)]
    del refcat

    # flag objects in the LMC and SMC
    for cloud in ['LMC', 'SMC']:
        cat[f'IN_{cloud}'] = find_in_mclouds(cat, mcloud=cloud)

    # flag objects in GCl / PNe
    cat['IN_GCLPNE'] = find_in_gclpne(cat)

    # apply cuts based on the photometry files
    #log.info('Processing the photometry files')
    photo, nphoto = [], 0
    for region in ['dr11-south', 'dr9-north']:
        photofiles = sorted(glob(os.path.join(sga_dir(), 'parent', 'photo', f'parent-photo-{region}-v?.?.fits')))
        for photofile in photofiles:
            photo1 = Table(fitsio.read(photofile))
            #log.info(f'Read {len(photo1):,d} objects from {photofile}')
            photo.append(photo1)
            nphoto += 1
    photo = vstack(photo)
    log.info(f'Read {len(photo):,d} objects from {nphoto} photometry files.')

    # Trim the photo catalog to the objects that are still in the sample.
    I = np.isin(photo['OBJNAME'], cat['OBJNAME'])
    log.info(f'Keeping {np.sum(I):,d}/{len(photo):,d} photometry rows of objects ' + \
          'that are still in the current parent sample.')
    photo = photo[I]

    # Do not throw out objects in the properties, actions, or
    # 'custom' catalogs, which were added by-hand!
    custom = read_custom_external()
    I = np.logical_or.reduce((np.isin(photo['OBJNAME'], props['objname_ned']),
                              np.isin(photo['OBJNAME'], custom['OBJNAME_NED']),
                              np.isin(photo['OBJNAME'], actions['objname'])))
    log.info(f'Removing {np.sum(I):,d}/{len(photo):,d} photometry rows of ' + \
          'objects in the custom or properties tables.')
    photo = photo[~I]

    # Keep all small objects (with a low redshift) in and around the
    # Coma and Virgo clusters.
    virgo = (187.705833, 12.391111, 5.5) # radius in degrees
    coma = (194.898750, 27.959167, 1.5)
    for name, (racen, deccen, rad_degrees) in zip(['Coma', 'Virgo'], [coma, virgo]):
        m1, m2, _ = match_radec(photo['RA'], photo['DEC'], racen, deccen, rad_degrees)
        clust = join(cat['OBJNAME', 'RA', 'DEC', 'Z', 'DIAM_LIT', 'BA_LIT', 'PA_LIT'], photo['OBJNAME', ][m1], keys='OBJNAME')
        clust = clust[(clust['Z'] != -99.) * (clust['Z'] < 0.1)] # low-z
        log.info(f'Removing {len(clust):,d}/{len(photo):,d} objects with z<0.1 around the {name} cluster.')
        photo = photo[~np.isin(photo['OBJNAME'], clust['OBJNAME'])]

        if True:
            clust.rename_columns(['OBJNAME', 'RA', 'DEC', 'DIAM_LIT', 'BA_LIT', 'PA_LIT'],
                                 ['name', 'ra', 'dec', 'radius', 'abratio', 'posAngle'])
            clust['posAngle'][clust['posAngle'] < 0.] = 0.
            clust['abratio'][clust['abratio'] < 0.] = 1.
            I = clust['radius'] < 20./60.
            if np.any(I):
                clust['radius'][I] = 20./60.
            clust['radius'] *= 60. / 2.
            clust.write(f'viewer-{name}.fits', overwrite=True)

    # FIXME - we may want to keep more of these...

    # The photo files are limited to objects with DIAM_INIT<20 arcsec;
    # after significant VI, most of these are genuinely small objects,
    # so throw them all out for now. However, Keep objects in the
    # LMC,SMC, since those were also visually inspected.
    I = np.isin(cat['OBJNAME'], photo['OBJNAME'])
    J = np.logical_or(~I, np.logical_or(cat['IN_LMC'], cat['IN_SMC']))
    log.info(f'Removing {np.sum(J):,d}/{len(cat):,d} objects with diam_init<20 ' + \
          'arcsec (not in the SMC,LMC) based on the photometry files.')
    cat = cat[J]

    lvdmiss = check_lvd(cat)
    if lvdmiss is not None:
        log.info(lvdmiss)
        pdb.set_trace()

    # For convenience, add dedicated Boolean columns for each external
    # file (which may be different than the ELLIPSEMODE and SAMPLE bits
    # which will be populated in build_parent).
    for action in ['fixgeo', 'resolved', 'forcepsf', 'lessmasking', 'moremasking']:
        actfile = resources.files('SGA').joinpath(f'data/SGA2025/SGA2025-{action}.csv')
        if not os.path.isfile(actfile):
            log.warning(f'No action file {actfile} found; skipping.')
            continue

        # read the file and check for duplicates
        act = Table.read(actfile, format='csv', comment='#')
        log.info(f'Read {len(act)} objects from {actfile}')

        oo, cc = np.unique(act['objname'].value, return_counts=True)
        if np.any(cc > 1):
            log.warning(f'duplicates in action file {actfile}')
            log.info(oo[cc>1])
            pdb.set_trace()

        # make sure every object is in the current catalog
        I = np.isin(cat['OBJNAME'].value, act['objname'].value)
        if np.sum(I) != len(act):
            log.warning(f'The parent catalog is missing the following objects in {actfile}')
            log.info(act[~np.isin(act['objname'].value, cat['OBJNAME'].value)])
            pdb.set_trace()

        # finally add a Boolean flag
        col = f'IN_{action.upper()}'
        cat[col] = np.zeros(len(cat), bool)
        cat[col][I] = True

    # RESOLVED always implies FIXGEO!
    cat['IN_FIXGEO'][cat['IN_RESOLVED']] = True

    # add the Gaia masking bits
    add_gaia_masking(cat)

    log.info(f'Writing {len(cat):,d} objects to {final_outfile}')
    cat.meta['EXTNAME'] = 'PARENT-ARCHIVE'
    cat.write(final_outfile, overwrite=True)


def update_geometry_from_reffiles(parent, diam, ba, pa, diam_ref, reffiles,
                                  REGIONBITS, veto_objnames=None,
                                  region_order=['dr9-north', 'dr11-south']):
    # optionally update initial diameters
    ref_version = reffiles['ref_version']
    log.info(f'Updating initial diameters using reference version {ref_version}')

    for region in region_order:  # dr11-south wins in overlaps
        bit = REGIONBITS[region]
        reffile = reffiles[region]

        ref_tab = Table(fitsio.read(reffile, 'ELLIPSE', columns=[
            'OBJNAME', 'REGION', 'RA', 'DEC', 'GROUP_MULT',
            'BA', 'PA', 'D26', 'D26_REF']))
        if ref_version == 'v0.11':
            # not all ellipse diameters in v0.11 were reliable
            ref_tab = ref_tab[ref_tab['GROUP_MULT'] == 1]
        log.info(f'Read {len(ref_tab):,d} objects from {reffile}')

        # match by OBJNAME
        m_parent, m_ref = match(parent['OBJNAME'], ref_tab['OBJNAME'])

        # only objects whose REGION includes this bit
        keep = (parent['REGION'][m_parent] & bit) != 0
        if veto_objnames is not None:
            not_veto  = ~np.isin(parent['OBJNAME'][m_parent], list(veto_objnames))
            keep = keep & not_veto

        idx_p = m_parent[keep]
        idx_r = m_ref[keep]

        # reference geometry
        ref_diam = ref_tab['D26'][idx_r].value # [arcmin]
        ref_ba = ref_tab['BA'][idx_r].value
        ref_pa = ref_tab['PA'][idx_r].value
        ref_ref = ref_version + '/' + ref_tab['D26_REF'][idx_r].value

        diam[idx_p] = ref_diam # [arcsec]
        ba[idx_p] = ref_ba
        pa[idx_p] = ref_pa
        diam_ref[idx_p] = ref_ref

        # update coordinates?

    return diam, ba, pa, diam_ref


def build_parent_legacy(mp=1, reset_sgaid=False, verbose=False, overwrite=False):
    """Build the parent catalog.

    """
    from astropy.table import Column
    from desiutil.dust import SFDMap
    from SGA.geometry import choose_geometry, in_ellipse_mask, ellipses_overlap
    from SGA.SGA import sga2025_name, SAMPLE, SGA_version
    from SGA.ellipse import ELLIPSEMODE, FITMODE, ELLIPSEBIT
    from SGA.io import radec_to_groupname
    from SGA.groups import build_group_catalog, make_singleton_group
    from SGA.coadds import REGIONBITS
    from SGA.sky import find_close, in_ellipse_mask_sky


    version = SGA_version(parent=True)
    version_vicuts = SGA_version(vicuts=True)
    version_nocuts = SGA_version(nocuts=True)
    version_archive = SGA_version(archive=True)
    parentdir = os.path.join(sga_dir(), 'parent')
    outdir = os.path.join(sga_dir(), 'sample')

    outfile = os.path.join(outdir, f'SGA2025-parent-{version}.fits')
    kdoutfile = os.path.join(outdir, f'SGA2025-parent-{version}.kd.fits')
    if os.path.isfile(outfile) and not overwrite:
        log.info(f'Parent catalog {outfile} exists; use --overwrite')
        return

    cols = ['OBJNAME', 'RA', 'DEC', 'PGC', 'REGION']

    # merge the two regions
    parent, lvd_dwarfs = [], []
    for region in ['dr11-south', 'dr9-north']:
        catfile = os.path.join(parentdir, f'SGA2025-parent-archive-{region}-{version_archive}.fits')
        cat = Table(fitsio.read(catfile))#, rows=np.arange(5000)))
        log.info(f'Read {len(cat):,d} objects from {catfile}')

        lvd_dwarfs.append(cat['OBJNAME'][cat['ROW_LVD'] != -99].value)

        # add the region bit
        cat['REGION'] = np.int16(REGIONBITS[region])
        parent.append(cat)

    parent = vstack(parent)
    lvd_dwarfs = np.unique(np.hstack(lvd_dwarfs))

    # merge north-south duplicates
    parent.remove_columns(['NCCD', 'FILTERS']) # can be useful to see in the duplicates
    dups, cc = np.unique(parent['OBJNAME'].value, return_counts=True)
    dup = parent[np.isin(parent['OBJNAME'], dups[cc>1])]
    dup = dup[np.argsort(dup['OBJNAME'])]
    assert(np.all(dup['OBJNAME'][0::2] == dup['OBJNAME'][1::2]))
    log.info(f'Found {len(dup):,d}/{len(parent):,d} objects ' + \
             'in north-south overlap region.')

    parent = parent[~np.isin(parent['OBJNAME'], dup['OBJNAME'])]

    dup = dup[0::2] # choose every other one
    dup['REGION'] = 2**0 + 2**1

    parent = vstack((parent, dup))
    parent = parent[np.lexsort([parent['OBJNAME'].value, parent['RA'].value])]
    log.info(f'Combined parent sample has {len(parent):,d} unique objects.')
    assert(np.sum(parent['REGION'] == 3) == len(dup) == len(dups[cc>1]))

    # make sure we haven't dropped any LVD dwarfs
    assert(np.all(np.isin(lvd_dwarfs, parent['OBJNAME'])))


    # add additional sources by-hand
    def _empty_parent(cat, N=1):
        """Return a new Table with same columns/dtypes as `cat`,
        length N, initialized with your null values.
        """
        empty = Table(masked=False)

        for col in cat.itercols():
            name = col.name
            dt = col.dtype
            if dt.kind == 'b':                      # bool
                data = np.zeros(N, dtype=dt)
            elif dt.kind in 'iu' or dt.kind == 'f': # int / uint and float
                fill = 99 if name in ['STARMAG', 'STARDIST', 'STARFDIST'] else -99
                data = np.full(N, fill, dtype=dt)
            elif dt.kind in 'SU':                   # fixed-length string
                data = np.full(N, '', dtype=dt)
            else:
                data = np.full(N, -99, dtype=dt)
            empty[name] = Column(data, name=name, dtype=dt)
        return empty


    customfile = resources.files('SGA').joinpath(f'data/SGA2025/SGA2025-parent-custom.csv')
    custom = Table.read(customfile, format='csv', comment='#')
    log.info(f'Read {len(custom)} objects from {customfile}')
    try:
        assert(len(custom) == len(np.unique(custom['OBJNAME'])))
    except:
        log.info('Warning: duplicates in parent-custom file!')
        oo, cc = np.unique(custom['OBJNAME'], return_counts=True)
        log.info(oo[cc>1])
        raise ValueError()

    # special handling of SREGION; if masked then it's both regions
    if hasattr(custom['SREGION'], 'mask'):
        custom['SREGION'] = custom['SREGION'].filled('')
    custom['REGION'] = np.zeros(len(custom), dtype=parent['REGION'].dtype)
    for iobj, reg in enumerate(custom['SREGION']):
        if reg == '':
            custom['REGION'][iobj] = REGIONBITS['dr11-south'] + REGIONBITS['dr9-north']
        else:
            custom['REGION'][iobj] = REGIONBITS[reg]
    custom.remove_columns(['SREGION', 'COMMENT'])

    moreparent = _empty_parent(parent[:1], len(custom))
    for col in custom.colnames:
        if col == 'COMMENT':
            continue
        moreparent[col] = np.asarray(custom[col], dtype=parent[col].dtype)
    #moreparent[custom.colnames]

    # update the Gaia masking bits
    add_gaia_masking(moreparent)

    # Some objects were dropped in the 'archive' step but restored via
    # the custom file; restore their properties here.
    nocuts_objnames = fitsio.read(os.path.join(parentdir, f'SGA2025-parent-nocuts-{version_nocuts}.fits'), columns='OBJNAME')
    rows = np.isin(nocuts_objnames, moreparent['OBJNAME'])
    if np.any(rows):
        rows = np.where(np.isin(nocuts_objnames, moreparent['OBJNAME']))[0]
        nocuts = Table(fitsio.read(os.path.join(parentdir, f'SGA2025-parent-nocuts-{version_nocuts}.fits'), rows=rows))
        m_parent, m_nocuts = match(moreparent['OBJNAME'], nocuts['OBJNAME'])
        for col in nocuts.colnames:
            if col in custom.colnames: # do not overwrite the custom file
                continue
            moreparent[col][m_parent] = nocuts[col][m_nocuts]

    # assign a unique parent_row for new objects
    I = np.where(moreparent['ROW_PARENT'] < 0)[0]
    if len(I) > 0:
        all_parent_rows = fitsio.read(os.path.join(parentdir, f'SGA2025-parent-nocuts-{version_nocuts}.fits'), columns='ROW_PARENT')
        moreparent['ROW_PARENT'][I] = np.max(all_parent_rows) + np.arange(len(I)) + 1

    #####################
    ## hack!
    #keep_in_custom_objname = moreparent[~np.isin(moreparent['ROW_PARENT'], parent['ROW_PARENT'])]['OBJNAME']
    #custom = Table.read(customfile, format='csv', comment='#')
    #keep_in_custom = custom[np.isin(custom['OBJNAME'], keep_in_custom_objname)]
    #keep_in_custom.write('keep_in_custom.csv', format='csv', overwrite=True)
    #
    #move_to_properties = moreparent[np.isin(moreparent['ROW_PARENT'], parent['ROW_PARENT'])]
    #move_to_properties['COMMENT'] = 'Moved from custom'
    #move_to_properties['OBJNAME', 'RA', 'DEC', 'DIAM_LIT', 'PA_LIT', 'BA_LIT', 'COMMENT'].write('move-to-properties.csv', format='csv', overwrite=True)

    parent = vstack((parent, moreparent))
    if len(parent) != len(np.unique(parent['ROW_PARENT'])):
        log.info('Duplicate ROW_PARENT values between parent and moreparent!')
        oo, cc = np.unique(parent['ROW_PARENT'], return_counts=True)
        log.info(oo[cc>1])
        bb = parent[np.isin(parent['ROW_PARENT'], oo[cc>1])]['OBJNAME', 'ROW_PARENT'] ; bb = bb[np.argsort(bb['ROW_PARENT'])] ; bb
        pdb.set_trace()

    assert(len(parent) == len(np.unique(parent['ROW_PARENT'])))

    # Read and process the "parent-drop" file; update REGION for
    # objects indicated in that file.
    dropfile = resources.files('SGA').joinpath(f'data/SGA2025/SGA2025-parent-drop.csv')
    drop = Table.read(dropfile, format='csv', comment='#')
    log.info(f'Read {len(drop)} objects from {dropfile}')
    try:
        assert(len(drop) == len(np.unique(drop['objname'])))
    except:
        log.info('Warning: duplicates in parent-drop file!')
        #raise ValueError()
        oo, cc = np.unique(drop['objname'], return_counts=True)
        log.info(oo[cc>1])
        pdb.set_trace()

    # drop crap from both/all regions
    Idrop = drop['region'].mask
    log.info(f'Dropping {np.sum(Idrop):,d}/{len(parent):,d} objects based on VI')
    parent = parent[~np.isin(parent['OBJNAME'], drop['objname'][Idrop])]

    # drop crap for specified regions
    drop = drop[~drop['region'].mask]
    for region in ['dr11-south', 'dr9-north']:
        bit = REGIONBITS[region]
        Idrop = drop['region'] == region
        Iregion = (parent['REGION'] & bit != 0)
        Iparent = Iregion * np.isin(parent['OBJNAME'], drop['objname'][Idrop])
        log.info(f'Dropping {np.sum(Idrop):,d}/{np.sum(Iregion):,d} objects in {region} based on VI')
        parent['REGION'][Iparent] -= bit

    # totally out of the sample...
    I = parent['REGION'] == 0
    if np.any(I):
        parent = parent[~I]

    mindiam = 20.
    diam, ba, pa, diam_ref, mag, band = choose_geometry(
        parent, mindiam=mindiam, get_mag=True)
    origdiam, _, _, _ = choose_geometry(parent, mindiam=0.)

    # cleanup magnitudes
    band[band == ''] = 'V' # default
    I = (mag > 20.) | (mag < 0.)
    if np.any(I):
        mag[I] = 20.

    # Restore diameters for LVD sources which have diam<mindiam (e.g.,
    # Clump I)
    I = (parent['ROW_LVD'] != -99) * (diam <= mindiam)
    log.info(f'Restoring {np.sum(I)} LVD diameters that are <mindiam={mindiam:.1f} arcsec')
    diam[I] = origdiam[I]

    I = diam == mindiam
    if np.any(I):
        log.warning(f'Setting mindiam={mindiam:.1f} arcsec for {np.sum(I):,d} objects.')
    diam /= 60. # [arcmin]
    assert(np.all(diam > 0.))


    # use "reference" diameters
    reffiles = {
        # The v0.12 parent catalog was constructed by updating the
        # initial diameters from the v0.11 values.
        'v0.12': {
            'ref_version': 'v0.11',
            'dr9-north': os.path.join(outdir, 'SGA2025-v0.11-dr9-north.fits'),
            'dr11-south': os.path.join(outdir, 'SGA2025-v0.11-dr11-south.fits'),
            },
        # In v0.20-v0.22 use the v0.11 measurements to throw out small galaxies.
        'v0.20': {
            'ref_version': 'v0.11',
            'dr9-north': os.path.join(outdir, 'SGA2025-v0.11-dr9-north.fits'),
            'dr11-south': os.path.join(outdir, 'SGA2025-v0.11-dr11-south.fits'),
            },
        'v0.21': {
            'ref_version': 'v0.11',
            'dr9-north': os.path.join(outdir, 'SGA2025-v0.11-dr9-north.fits'),
            'dr11-south': os.path.join(outdir, 'SGA2025-v0.11-dr11-south.fits'),
            },
        'v0.22': {
            'ref_version': 'v0.11',
            'dr9-north': os.path.join(outdir, 'SGA2025-v0.11-dr9-north.fits'),
            'dr11-south': os.path.join(outdir, 'SGA2025-v0.11-dr11-south.fits'),
            },
    }
    if version == 'v0.12':
        # in v0.11 the 'fixgeo' geometry was inadvertently
        # overwritten, so don't update those diameters
        veto = []
        for action in ['fixgeo', 'resolved']:
            actfile = resources.files('SGA').joinpath(f'data/SGA2025/SGA2025-{action}.csv')
            veto.append(Table.read(actfile, format='csv', comment='#'))
        veto = vstack(veto)
        veto_objnames = set(np.unique(veto['objname'].tolist()))

        diam, ba, pa, diam_ref = update_geometry_from_reffiles(
            parent, diam, ba, pa, diam_ref, reffiles[version],
            REGIONBITS, veto_objnames=veto_objnames)
    elif version == 'v0.20' or version == 'v0.21' or version == 'v0.22':
        # Apply D(26)>0.5 diameter cuts but do not drop objects from the
        # "properties" or "custom" catalog.
        propsfile = resources.files('SGA').joinpath(f'data/SGA2025/SGA2025-parent-properties.csv')
        customfile = resources.files('SGA').joinpath(f'data/SGA2025/SGA2025-parent-custom.csv')
        props = Table.read(propsfile, format='csv', comment='#')
        custom = Table.read(customfile, format='csv', comment='#')

        veto_objnames = np.unique(np.hstack([props['objname'], custom['OBJNAME']]))
        veto_objnames = set(veto_objnames.tolist())

        for region in ['dr9-north', 'dr11-south']:
            reffile = reffiles[version][region]
            ref_tab = Table(fitsio.read(reffile, 'ELLIPSE', columns=[
                'OBJNAME', 'REGION', 'SAMPLE', 'ELLIPSEBIT', 'RA', 'DEC',
                'GROUP_NAME', 'GROUP_MULT', 'RA_INIT', 'DEC_INIT', 'DIAM_INIT',
                'BA', 'PA', 'D26', 'D26_REF']))
            parent, _ = remove_small_groups_and_galaxies(
                parent, ref_tab, region, REGIONBITS, SAMPLE,
                ELLIPSEBIT, mindiam=0.5, veto_objnames=veto_objnames)

        # Drop objects with REGION==0.
        I = parent['REGION'] != 0
        parent = parent[I]
        diam = diam[I]
        ba = ba[I]
        pa = pa[I]
        diam_ref = diam_ref[I]
        mag = mag[I]
        band = band[I]
        assert(np.all(np.isin(lvd_dwarfs, parent['OBJNAME'])))

    # one final update of coordinates and geometry based on VI
    propsfile = resources.files('SGA').joinpath(f'data/SGA2025/SGA2025-parent-properties.csv')
    props = Table.read(propsfile, format='csv', comment='#')
    log.info(f'Read {len(props)} objects from {propsfile}')
    try:
        assert(len(props) == len(np.unique(props['objname'])))
    except:
        log.info('Warning: duplicates in parent-properties file!')
        #raise ValueError()
        oo, cc = np.unique(props['objname'], return_counts=True)
        log.info(oo[cc>1])
        pdb.set_trace()

    miss = props[~np.isin(props['objname'], parent['OBJNAME'].value)]
    if len(miss) > 0:
        log.info(f'The following objects in {propsfile} are missing from parent:')
        log.info(miss)
        pdb.set_trace()

    for prop in props:
        objname = prop['objname']
        I = np.where(objname == parent['OBJNAME'].value)[0]
        for col in ['ra', 'dec', 'diam', 'pa', 'ba']:
            newval = prop[col]
            if col == 'ra' or col == 'dec':
                oldval = parent[col.upper()][I[0]]
            elif col == 'diam':
                oldval = diam[I[0]]
            elif col == 'pa':
                oldval = pa[I[0]]
            elif col == 'ba':
                oldval = ba[I[0]]

            if newval != -99.:
                pass
                #log.info(f'{objname} {col}: {oldval} --> {newval}')
            else:
                pass
                #log.info(f'  Retaining {col}: {oldval}')

            if newval != -99.:
                if col == 'ra' or col == 'dec':
                    parent[col.upper()][I] = newval
                elif col == 'diam':
                    diam[I] = newval
                elif col == 'pa':
                    pa[I] = newval
                elif col == 'ba':
                    ba[I] = newval

    # Also re-update geometry based on the custom file since the
    # values can get overwritten by choose_diameter (e.g., KUG
    # 1206+425 has an SGA-2020 diameter that we don't want).
    log.info('One more geometry update using the customfile.')
    customfile = resources.files('SGA').joinpath(f'data/SGA2025/SGA2025-parent-custom.csv')
    custom = Table.read(customfile, format='csv', comment='#')
    for cust in custom:
        objname = cust['OBJNAME']
        I = np.where(objname == parent['OBJNAME'].value)[0]
        if len(I) == 0:
            pdb.set_trace()
        else:
            for col in ['RA', 'DEC', 'DIAM_LIT', 'PA_LIT', 'BA_LIT']:
                newval = cust[col]
                if col == 'RA' or col == 'DEC':
                    oldval = parent[col.upper()][I[0]]
                elif col == 'DIAM_LIT':
                    oldval = diam[I[0]]
                elif col == 'PA_LIT':
                    oldval = pa[I[0]]
                elif col == 'BA_LIT':
                    oldval = ba[I[0]]

                if newval != -99. and newval != oldval:
                    log.info(f'{objname} {col}: {oldval} --> {newval}')
                    pass
                else:
                    pass
                    #log.info(f'  Retaining {col}: {oldval}')

                if newval != -99. and newval != oldval:
                    if col == 'RA' or col == 'DEC':
                        parent[col.upper()][I] = newval
                    elif col == 'DIAM_LIT':
                        diam[I] = newval
                    elif col == 'PA_LIT':
                        pa[I] = newval
                    elif col == 'BA_LIT':
                        ba[I] = newval


    # Assign the SAMPLE bits.
    samplebits = np.zeros(len(parent), np.int32)
    samplebits[parent['ROW_LVD'] != -99] |= SAMPLE['LVD']       # 2^0 - LVD dwarfs
    samplebits[parent['STARFDIST'] < 1.2] |= SAMPLE['NEARSTAR'] # 2^3 - NEARSTAR
    samplebits[parent['STARFDIST'] < 0.5] |= SAMPLE['INSTAR']   # 2^4 - INSTAR

    # Re-populate the LMC/SMC and GCLPNE bits since we've added objects.
    for cloud in ['LMC', 'SMC']:
        in_mcloud = find_in_mclouds(parent, mcloud=cloud)
        samplebits[in_mcloud] |= SAMPLE['MCLOUDS'] # 2^1 - Magellanic Clouds

    in_gclpne = find_in_gclpne(parent)
    samplebits[in_gclpne] |= SAMPLE['GCLPNE']      # 2^2 - GC/PNe

    # Assign the ELLIPSEMODE bits. Rederive the set of objects in each
    # file so those files can be updated without having to rerun
    # build_parent_archive.
    ellipsemode = np.zeros(len(parent), np.int32)
    actions = ['fixgeo', 'resolved', 'forcepsf', 'lessmasking', 'moremasking',
               'momentpos', 'tractorgeo', 'radweight']
    for action in actions:
        actfile = resources.files('SGA').joinpath(f'data/SGA2025/SGA2025-{action}.csv')
        if not os.path.isfile(actfile):
            log.warning(f'No action file {actfile} found; skipping.')
            continue

        # read the file and check for duplicates
        act = Table.read(actfile, format='csv', comment='#')
        log.info(f'Read {len(act)} objects from {actfile}')

        oo, cc = np.unique(act['objname'].value, return_counts=True)
        if np.any(cc > 1):
            log.warning(f'duplicates in action file {actfile}')
            log.info(oo[cc>1])
            raise ValueError()

        # make sure every object is in the current catalog
        I = np.isin(parent['OBJNAME'].value, act['objname'].value)
        if np.sum(I) != len(act):
            log.warning(f'The parent catalog is missing the following objects in {actfile}')
            log.info(act[~np.isin(act['objname'].value, parent['OBJNAME'].value)])
            raise ValueError()

        ellipsemode[I] |= ELLIPSEMODE[action.upper()]

    # RESOLVED always implies FIXGEO!
    ellipsemode[ellipsemode & ELLIPSEMODE['RESOLVED'] != 0] |= ELLIPSEMODE['FIXGEO']

    # FITMODE is used by legacypipe
    fitmode = np.zeros(len(parent), np.int32)
    fitmode[ellipsemode & ELLIPSEMODE['FIXGEO'] != 0] |= FITMODE['FIXGEO']
    fitmode[ellipsemode & ELLIPSEMODE['RESOLVED'] != 0] |= FITMODE['RESOLVED']

    # build the final catalog.
    grp = parent['REGION', 'OBJNAME', 'PGC']
    grp['SAMPLE'] = samplebits
    grp['ELLIPSEMODE'] = ellipsemode
    grp['FITMODE'] = fitmode
    grp['RA'] = parent['RA']
    grp['DEC'] = parent['DEC']
    grp['DIAM'] = diam.astype('f4') # [arcmin]
    grp['BA'] = ba.astype('f4')
    grp['PA'] = (pa % 180.).astype('f4')
    grp['MAG'] = mag.astype('f4')
    #grp['MAG_BAND'] = band
    grp['DIAM_REF'] = diam_ref

    ## apply an additional diameter cut:
    #if float(version[1:]) >= 0.13:
    #    print('APPLY VI CUTS!')

    # Add SFD dust
    SFD = SFDMap(scaling=1.0)
    #grp.add_column(SFD.ebv(grp['RA'].value, grp['DEC'].value), name='EBV', index=10)
    grp['EBV'] = SFD.ebv(grp['RA'].value, grp['DEC'].value).astype('f4')

    log.info('Reverse-sorting by diameter')
    srt = np.argsort(diam)[::-1]
    grp = grp[srt]

    if reset_sgaid:
        log.info('Resetting SGAID')
        sgaid = np.arange(len(grp))
    else:
        log.info('Adopting ROW_PARENT for SGAID')
        sgaid = parent['ROW_PARENT'][srt].value
    grp.add_column(sgaid, name='SGAID', index=0)
    assert(len(grp) == len(np.unique(grp['SGAID'])))


    # Build the group catalog but make sure the RESOLVED and FORCEPSF
    # samples (e.g., SMC, LMC) are alone.
    I = np.logical_or((grp['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED']) != 0,
                      (grp['ELLIPSEMODE'] & ELLIPSEMODE['FORCEPSF']) != 0)
    out1 = make_singleton_group(grp[I], group_id_start=0)
    #out2 = build_group_catalog(grp[~I], group_id_start=max(out1['GROUP_ID'])+1, mp=mp)
    out2 = build_group_catalog(grp[~I], group_id_start=max(out1['GROUP_ID'])+1, mp=mp)
    out = vstack((out1, out2))
    #del out1, out2

    # assign SGAGROUP from GROUP_NAME and check for duplicates
    groupname = np.char.add('SGA2025_', out['GROUP_NAME'])
    out.add_column(groupname, name='SGAGROUP', index=1)

    I = out['GROUP_PRIMARY']
    gg, cc = np.unique(out['SGAGROUP'][I], return_counts=True)
    if len(gg[cc>1]) > 0:
        print('Duplicate groups!!')
        pdb.set_trace()

    # After assigning groups, loop back through and make sure REGION
    # is the same for all group members, otherwise SGA.build_catalog
    # will think that a galaxy is missing for reasons other than there
    # are no data.
    # bits
    I = (out['GROUP_PRIMARY']) & (out['GROUP_MULT'] > 1)
    drop_groupid = []
    strip_groups = []
    for groupid in out['GROUP_ID'][I]:
        J = (out['GROUP_ID'] == groupid)
        # Bits common to ALL members of this group:
        allowed = int(np.bitwise_and.reduce(out['REGION'][J]))
        if allowed == 0:
            # no region bit shared by all members → drop entire group
            drop_groupid.append(groupid)
        else:
            # clear missing bits from everyone in the group (keep only the common bits)
            new_reg = (out['REGION'][J] & allowed)
            if np.any(new_reg != out['REGION'][J]):
                out['REGION'][J] = new_reg
                strip_groups.append(out['GROUP_NAME'][J][0])

    if drop_groupid:
        M = np.isin(out['GROUP_ID'], drop_groupid)
        log.info(f"Dropping {len(np.unique(drop_groupid)):,d} unique groups "
                 f"({np.sum(M):,d} members) with no common region bit.")
        out = out[~M]

    if len(strip_groups) > 0:
        strip_groups = np.unique(strip_groups)
        log.info(f"Stripped region bits (kept groups) for {len(strip_groups):,d} groups:")
        log.info(f"  {','.join(strip_groups)}")

    # For each unique group, assign the OVERLAP sample bit.
    set_overlap_bit(out, SAMPLE)

    # one more check!
    try:
        assert(np.all(np.isin(lvd_dwarfs, out['OBJNAME'])))
    except:
        pdb.set_trace()

    log.info(f'Writing {len(out):,d} objects to {outfile}')
    out.meta['EXTNAME'] = 'PARENT'
    out.write(outfile, overwrite=True)

    cmd = f'startree -i {outfile} -o {kdoutfile} -T -P -k -n stars'
    log.info(cmd)
    _ = os.system(cmd)


    ## Quick check that we have all LVD dwarfs: Yes! 623 (81) LVD
    ## objects within (outside) the DR11 imaging footprint.
    #import matplotlib.pyplot as plt
    #from SGA.external import read_lvd
    #lvd = read_lvd(verbose=False)
    #
    #lvdcat = out[out['SAMPLE'] & SAMPLE['LVD'] != 0]
    #lvdmiss = check_lvd(lvdcat=lvdcat)
    #fig, ax = plt.subplots(figsize=(8, 6))
    #ax.scatter(out['RA'], out['DEC'], s=1)
    #ax.scatter(lvdmiss['RA'], lvdmiss['DEC'], s=20, alpha=0.5, marker='s')
    #fig.savefig('ioannis/tmp/junk.png')
    #
    #nn = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/ioannis/SGA/2025/parent/SGA2025-parent-archive-dr9-north-v0.1.fits'))
    #ss = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/ioannis/SGA/2025/parent/SGA2025-parent-archive-dr11-south-v0.1.fits'))
    #nnlvdmiss = check_lvd(nn)
    #sslvdmiss = check_lvd(ss)
    #miss = lvd[np.isin(lvd['OBJNAME'], np.intersect1d(nnlvdmiss['OBJNAME'], sslvdmiss['OBJNAME']))]
    #
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    #ax1.scatter(nn['RA'], nn['DEC'], s=1)
    #ax1.scatter(nnlvdmiss['RA'], nnlvdmiss['DEC'], s=20, alpha=0.5, marker='s')
    #ax1.scatter(miss['RA'], miss['DEC'], s=20, alpha=0.5, marker='x', color='k')
    #ax2.scatter(ss['RA'], ss['DEC'], s=1)
    #ax2.scatter(sslvdmiss['RA'], sslvdmiss['DEC'], s=20, alpha=0.5, marker='s')
    #ax2.scatter(miss['RA'], miss['DEC'], s=20, alpha=0.5, marker='x', color='k')
    #fig.savefig('ioannis/tmp/junk.png')


Overlays = namedtuple('Overlays', 'adds updates drops flags')


def _require_columns(tab, required, name):
    """Raise if any required column is missing."""
    missing = [c for c in required if c not in tab.colnames]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def _read_csv_if_exists(path, required=None, name='table'):
    """Read CSV if present; return empty Table with required columns otherwise."""
    if not os.path.isfile(path):
        tab = Table()
        if required:
            for c in required:
                # heuristic dtypes for an empty shell table
                if c in ('OBJNAME', 'FIELD', 'COLUMN', 'REASON'):
                    tab[c] = np.array([], dtype='U64')
                else:
                    tab[c] = np.array([], dtype=float)
        return tab
    tab = Table.read(path, format='csv', comment='#')
    if required:
        _require_columns(tab, required, name)
    return tab


def load_overlays(overlay_dir):
    """
    Load overlays from overlay_dir and return a container with
    .adds, .updates, .drops, .flags (each an Astropy Table, possibly empty).
    """
    adds = _read_csv_if_exists(overlay_dir / 'adds.csv', required=('OBJNAME', 'REGION', 'RA', 'DEC', 'DIAM', 'BA', 'PA'), name='adds.csv')
    updates = _read_csv_if_exists(overlay_dir / 'updates.csv', required=('OBJNAME', 'FIELD', 'NEW_VALUE'), name='updates.csv')
    drops = _read_csv_if_exists(overlay_dir / 'drops.csv', required=('OBJNAME', 'REGION'), name='drops.csv')
    flags = _read_csv_if_exists(overlay_dir / 'flags.csv', required=('target', 'value', 'column', 'op', 'bits'), name='flags.csv')

    # simple sanity checks
    if len(np.unique(adds['OBJNAME'])) != len(adds):
        log.critical("adds.csv: duplicate OBJNAME entries are not allowed.")
        oo, cc = np.unique(adds['OBJNAME'], return_counts=True)
        print(oo[cc>1])
        raise ValueError()
    if len(np.unique(drops['OBJNAME'])) != len(drops):
        log.critical("drops.csv: duplicate OBJNAME entries are not allowed.")
        oo, uindx, cc = np.unique(drops['OBJNAME'].value, return_counts=True, return_index=True)
        uindx.sort()
        drops[uindx].write('udrops.csv', format='csv', overwrite=True)
        log.warning(f'Wrote udrops.csv with {len(uindx):,d}/{len(drops):,d} unique rows.')
        raise ValueError()

    # normalize optional columns
    if 'REASON' not in updates.colnames:
        updates['REASON'] = np.array([], dtype='U64')

    return Overlays(adds=adds, updates=updates, drops=drops, flags=flags)


def apply_updates_inplace(parent, updates):
    """For each (OBJNAME, FIELD, NEW_VALUE) row, set parent[FIELD] accordingly."""

    if len(updates) == 0:
        return parent
    log.info(f'Updating parameter values for {len(updates):,d} object(s) using updates.csv')

    for row in updates:
        obj = row['OBJNAME']
        fld = row['FIELD']
        idx = np.where(parent['OBJNAME'] == obj)[0]
        if idx.size == 0:
            #continue
            raise ValueError(f"updates: OBJNAME not found: {obj}")
        # cast to parent dtype
        dt = parent[fld].dtype
        val = row['NEW_VALUE']
        #log.info(f'{obj} {fld}: {parent[fld][idx][0]} --> {val}')
        if dt.kind in 'f':
            parent[fld][idx] = float(val)
        elif dt.kind in 'iu':
            parent[fld][idx] = int(round(float(val)))
        elif dt.kind in 'SU':
            parent[fld][idx] = str(val)
        else:
            parent[fld][idx] = val

    # keep PA in [0,180) if PA was touched at all
    if 'PA' in parent.colnames:
        parent['PA'] = (parent['PA'].astype(float) % 180.).astype(parent['PA'].dtype)


def apply_drops(parent, drops, REGIONBITS):

    names = np.asarray(parent['OBJNAME']).astype(str)
    reg = np.asarray(parent['REGION']).astype(np.int32).copy()

    for obj, rgn in zip(drops['OBJNAME'], drops['REGION']):
        idx = np.flatnonzero(names == str(obj))
        if idx.size == 0:
            continue
        s = '' if rgn is None else str(rgn).strip()
        try:
            if s:  # drop from one region
                reg[idx] &= ~int(REGIONBITS[s])
            else:   # blank -> drop entirely
                reg[idx] = 0
        except KeyError:
            # treat placeholders like '--' as blank
            reg[idx] = 0

    keep = reg != 0
    log.info(f'Removing {np.sum(~keep):,d}/{len(parent):,d} objects from drops.csv file')

    out = parent[keep]
    out['REGION'] = reg[keep].astype(parent['REGION'].dtype, copy=False)
    return out


def apply_adds(parent, adds, regionbits, nocuts):
    """
    Append rows from adds. If OBJNAME exists in `nocuts`, start from that row
    (restored properties); otherwise fill minimal defaults. Then overwrite with
    add-file fields (OBJNAME, RA, DEC, REGION, DIAM, BA, PA, [MAG]).

    """
    if len(adds) == 0:
        return parent
    log.info(f'Adding {len(adds):,d} objects from adds.csv')

    # next SGAID
    maxsgaid = np.max(parent['SGAID'])
    maxrow = np.max(nocuts['ROW_PARENT'])
    next_sgaid = int(max(maxsgaid, maxrow)) + 1
    #print(next_sgaid)

    def _empty_row():
        row = {}
        for c in parent.colnames:
            dt = parent[c].dtype
            if dt.kind in 'f':   row[c] = np.array([-99.0], dtype=dt)
            elif dt.kind in 'iu':row[c] = np.array([0], dtype=dt)
            elif dt.kind in 'SU':row[c] = np.array([''], dtype=dt)
            else:                row[c] = np.array([-99], dtype=dt)
        return Table(row)

    new_rows = []
    for add in adds:
        obj = add['OBJNAME']

        # reject if already present
        if np.any(parent['OBJNAME'] == obj):
            log.warning(f"adds: OBJNAME already in parent: {obj}")
            continue

        base = _empty_row()
        base['OBJNAME'][0] = str(obj)
        base['RA'][0] = float(add['RA'])
        base['DEC'][0] = float(add['DEC'])
        base['DIAM'][0] = float(add['DIAM'])
        base['DIAM_REF'][0] = 'VI'
        base['BA'][0] = float(add['BA'])
        base['PA'][0] = float(add['PA']) % 180.0

        # REGION: if masked, set both bits; else map the single key
        reg_val = add['REGION']
        if np.ma.is_masked(reg_val):
            base['REGION'][0] = int(regionbits['dr11-south']) | int(regionbits['dr9-north'])
        else:
            base['REGION'][0] = int(regionbits[str(reg_val).strip()])

        # add from nocuts
        noc_ix = np.where(nocuts['OBJNAME'] == obj)[0]
        if len(noc_ix) == 1:
            src = nocuts[noc_ix[0]]
            base['PGC'] = src['PGC']
            base['SGAID'] = src['ROW_PARENT']
        else:
            base['SGAID'][0] = next_sgaid
            next_sgaid += 1

        new_rows.append(base)
        #print(next_sgaid)

    if new_rows:
        parent = vstack([parent] + new_rows)

    try:
        assert(len(parent) == len(np.unique(parent['SGAID'])))
    except:
        msg = 'Non-unique SGAID values!'
        log.critical(msg)
        raise ValueError(msg)

    return parent


def apply_flags_inplace(parent, flags, ELLIPSEMODE):
    """
    Apply consolidated flags to `parent` in place.

    Parameters
    ----------
    parent : astropy.table.Table
        Must contain columns: 'OBJNAME', 'ELLIPSEMODE'
    flags : astropy.table.Table
        Consolidated flags table with columns:
        ['target','value','column','op','bits'], where:
          - target == 'OBJNAME'
          - value  == object name (string)
          - column in {'ELLIPSEMODE'}
          - op in {'set','clear'}
          - bits is comma-separated actions already consolidated
            (and already enforcing RESOLVED ⇒ FIXGEO).
    ELLIPSEMODE : dict
        Bit dictionaries with UPPER-CASE keys (e.g., {'FIXGEO': 1<<0, ...}).

    """
    from collections import defaultdict

    # Prebuild index by OBJNAME
    name_to_rows = defaultdict(list)
    objnames = np.asarray(parent['OBJNAME']).astype(str)
    for i, nm in enumerate(objnames):
        name_to_rows[nm].append(i)

    # Lowercase maps for CSV bit names
    ellipse_map = {k.lower(): v for k, v in ELLIPSEMODE.items()}
    def bits_to_mask(column, bits_str):
        bits = [b.strip().lower() for b in str(bits_str).split(',') if b.strip()]
        if column == 'ELLIPSEMODE':
            mapping = ellipse_map
        else:
            raise ValueError(f'Unknown column {column!r}')
        mask = 0
        for b in bits:
            try:
                mask |= mapping[b]
            except KeyError:
                raise KeyError(f'Unknown bit {b!r} for {column}')
        return mask

    # Apply row-by-row
    for row in flags:
        # Expect only OBJNAME targets
        target = str(row['target']).strip().upper()
        if target != 'OBJNAME':
            continue  # ignore anything else

        value  = str(row['value']).strip()
        column = str(row['column']).strip().upper()   # 'ELLIPSEMODE'
        op     = str(row['op']).strip().lower()       # 'set' or 'clear'
        mask   = bits_to_mask(column, row['bits'])

        idxs = name_to_rows.get(value, [])
        if not idxs:
            continue  # silently skip unknown names

        arr = parent[column]
        if op == 'set':
            arr[idxs] = arr[idxs] | mask
        elif op == 'clear':
            arr[idxs] = arr[idxs] & ~mask
        else:
            raise ValueError(f'Unknown op {op!r}')


def diagnose_drift(ell60, outdir, ref_version='v0.10'):
    """Compare v0.60 positions to original initial positions."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from SGA.ellipse import ELLIPSEBIT
    from SGA.SGA import SAMPLE

    # v0.60 current results
    #ell60 = Table(fitsio.read(os.path.join(outdir, 'SGA2025-beta-v0.60-dr11-south.fits')))

    # v0.22 should have the original initial values (or go to nocuts if needed)
    #ell22 = Table(fitsio.read(os.path.join(outdir, 'SGA2025-v0.10-dr11-south.fits')))
    #ell22 = Table(fitsio.read(os.path.join(outdir, 'SGA2025-v0.22-dr11-south.fits')))
    #I = (ell22['RA_INIT'] == 0.) & (ell22['DEC_INIT'] == 0.)
    #if np.any(I):
    #    ell22['RA_INIT'][I] = ell22['RA'][I]
    #    ell22['DEC_INIT'][I] = ell22['DEC'][I]

    if ref_version == 'v0.10':
        ell22 = Table(fitsio.read(os.path.join(outdir, 'SGA2025-parent-v0.10.fits')))
    else:
        ell22 = Table(fitsio.read(os.path.join(outdir, f'SGA2025-beta-parent-{ref_version}.fits')))

    for ver in ['v0.30', 'v0.40', 'v0.50', 'v0.60']:
        overlay_dir = resources.files('SGA').joinpath(f'data/SGA2025/overlays/{ver}')
        ov = load_overlays(overlay_dir)
        I = np.isin(ov.updates['OBJNAME'], ell22['OBJNAME'])
        ovup = ov.updates[I]
        apply_updates_inplace(ell22, ovup)

    ell22.rename_columns(['RA', 'DEC', 'DIAM', 'BA', 'PA', 'DIAM_REF'],
                         ['RA_INIT', 'DEC_INIT', 'DIAM_INIT', 'BA_INIT', 'PA_INIT', 'DIAM_INIT_REF'])

    # Match by OBJNAME
    ell60 = ell60.copy()
    m60, m22 = match(ell60['OBJNAME'], ell22['OBJNAME'])
    ell60 = ell60[m60]
    ell22 = ell22[m22]

    # Position drift from original initial to current measured
    # Original initial positions (from v0.22, which should be untouched)
    c_init = SkyCoord(ell22['RA_INIT']*u.deg, ell22['DEC_INIT']*u.deg)

    # Current measured positions
    c_now = SkyCoord(ell60['RA']*u.deg, ell60['DEC']*u.deg)

    pos_drift_arcsec = c_init.separation(c_now).to(u.arcsec).value

    # Diameter drift
    diam_init = ell22['DIAM_INIT']
    diam_now = ell60['D26']
    diam_ratio = diam_now / np.clip(diam_init, 0.01, None)

    # BA drift
    ba_init = ell22['BA_INIT']
    ba_now = ell60['BA']
    ba_diff = np.abs(ba_now - ba_init)

    # PA drift (wrapped)
    pa_init = ell22['PA_INIT']
    pa_now = ell60['PA']
    pa_diff = np.abs(((pa_now - pa_init + 90.0) % 180.0) - 90.0)

    # Summary statistics
    log.info(f"Position drift (arcsec): median={np.median(pos_drift_arcsec):.2f}, "
             f"90%={np.percentile(pos_drift_arcsec, 90):.2f}, "
             f"99%={np.percentile(pos_drift_arcsec, 99):.2f}, "
             f"max={np.max(pos_drift_arcsec):.2f}")

    log.info(f"Diameter ratio (new/init): median={np.median(diam_ratio):.2f}, "
             f"10%={np.percentile(diam_ratio, 10):.2f}, "
             f"90%={np.percentile(diam_ratio, 90):.2f}")

    out = Table({
        'OBJNAME': ell60['OBJNAME'],
        'SGAID': ell60['SGAID'],
        'RA': ell60['RA'],
        'DEC': ell60['DEC'],
        'GROUP_MULT': ell60['GROUP_MULT'],
        'ELLIPSEBIT': ell60['ELLIPSEBIT'],
        'IS_LVD': ell60['SAMPLE'] & SAMPLE['LVD'] != 0,
        'D26': ell60['D26'],
        'RA_ORIG': ell22['RA_INIT'],
        'DEC_ORIG': ell22['DEC_INIT'],
        'DIAM_ORIG': diam_init,
        'DIAM_ORIG_REF': ell22['DIAM_INIT_REF'],
        'PA_ORIG': pa_init,
        'BA_ORIG': ba_init,
        'pos_drift_arcsec': pos_drift_arcsec,
        'diam_ratio': diam_ratio,
        'ba_diff': ba_diff,
        'pa_diff': pa_diff,
    })

    return out


def flag_for_refit(diag, pos_thresh_arcsec=5.0, diam_ratio_lo=0.3, diam_ratio_hi=3.0):
    """Flag objects that likely have corrupted geometry."""
    from SGA.ellipse import ELLIPSEBIT

    # 1. Large position drift
    large_pos_drift = diag['pos_drift_arcsec'] > pos_thresh_arcsec

    # 2. Extreme diameter change
    extreme_diam = (diag['diam_ratio'] < diam_ratio_lo) | (diag['diam_ratio'] > diam_ratio_hi)

    # 3. Already flagged LARGESHIFT
    has_largeshift = (diag['ELLIPSEBIT'] & ELLIPSEBIT['LARGESHIFT']) != 0

    # 4. In a group (higher risk of contamination)
    in_group = diag['GROUP_MULT'] > 1

    # 5. NOTRACTOR or SKIPTRACTOR (known problems)
    has_notractor = (diag['ELLIPSEBIT'] & ELLIPSEBIT['NOTRACTOR']) != 0
    has_skiptractor = (diag['ELLIPSEBIT'] & ELLIPSEBIT['SKIPTRACTOR']) != 0

    # Combine criteria
    needs_refit = (
        large_pos_drift |
        extreme_diam |
        has_largeshift |
        has_notractor |
        has_skiptractor
    )

    # Higher scrutiny for groups
    group_suspicious = in_group & (
        (diag['pos_drift_arcsec'] > 2.0) |  # lower threshold for groups
        (diag['diam_ratio'] < 0.5) |
        (diag['diam_ratio'] > 2.0)
    )
    needs_refit |= group_suspicious

    log.info(f"Flagged for refit: {np.sum(needs_refit):,d}/{len(diag):,d}")
    log.info(f"  - Large position drift (>{pos_thresh_arcsec}\"): {np.sum(large_pos_drift):,d}")
    log.info(f"  - Extreme diameter change: {np.sum(extreme_diam):,d}")
    log.info(f"  - LARGESHIFT bit: {np.sum(has_largeshift):,d}")
    log.info(f"  - NOTRACTOR: {np.sum(has_notractor):,d}")
    log.info(f"  - SKIPTRACTOR: {np.sum(has_skiptractor):,d}")
    log.info(f"  - Suspicious groups: {np.sum(group_suspicious):,d}")

    # Check the catastrophic position drifts
    catastrophic = diag[diag['pos_drift_arcsec'] > 30.]  # > 1 arcmin
    log.info(f"Catastrophic drift (>30\"): {len(catastrophic):,d}")
    print(catastrophic['OBJNAME', 'SGAID', 'GROUP_MULT', 'pos_drift_arcsec', 'diam_ratio'][:20])

    # Breakdown of large position drift by group membership
    large_drift = diag['pos_drift_arcsec'] > 5.0
    log.info(f"Large drift in singles: {np.sum(large_drift & (diag['GROUP_MULT'] == 1)):,d}")
    log.info(f"Large drift in groups: {np.sum(large_drift & (diag['GROUP_MULT'] > 1)):,d}")

    # Overlap between categories
    log.info(f"LARGESHIFT AND large_drift: {np.sum((diag['ELLIPSEBIT'] & ELLIPSEBIT['LARGESHIFT'] != 0) & large_drift):,d}")
    log.info(f"LARGESHIFT but NOT large_drift: {np.sum((diag['ELLIPSEBIT'] & ELLIPSEBIT['LARGESHIFT'] != 0) & ~large_drift):,d}")

    in_group = diag['GROUP_MULT'] > 1
    group_pos_drift = in_group & (diag['pos_drift_arcsec'] > 2.0)
    group_diam_small = in_group & (diag['diam_ratio'] < 0.5)
    group_diam_large = in_group & (diag['diam_ratio'] > 2.0)

    log.info(f"Groups with pos drift >2\": {np.sum(group_pos_drift):,d}")
    log.info(f"Groups with diam_ratio <0.5: {np.sum(group_diam_small):,d}")
    log.info(f"Groups with diam_ratio >2.0: {np.sum(group_diam_large):,d}")

    # The diam_ratio > 2.0 in groups — are these mostly small initial diameters?
    group_large_ratio = (diag['GROUP_MULT'] > 1) & (diag['diam_ratio'] > 2.0)

    print(f"Median DIAM_INIT for group diam_ratio>2: {np.median(diag['DIAM_ORIG'][group_large_ratio]):.2f}")
    print(f"Median D26 for group diam_ratio>2: {np.median(diag['D26'][group_large_ratio]):.2f}")

    # Breakdown by initial size
    small_init = diag['DIAM_ORIG'] < 0.5
    med_init = (diag['DIAM_ORIG'] >= 0.5) & (diag['DIAM_ORIG'] < 1.0)
    large_init = diag['DIAM_ORIG'] >= 1.0

    log.info(f"Groups diam_ratio>2, DIAM_INIT<0.5': {np.sum(group_large_ratio & small_init):,d}")
    log.info(f"Groups diam_ratio>2, DIAM_INIT 0.5-1': {np.sum(group_large_ratio & med_init):,d}")
    log.info(f"Groups diam_ratio>2, DIAM_INIT>1': {np.sum(group_large_ratio & large_init):,d}")

    # Check position drift for these — if position also moved, more suspicious
    log.info(f"Groups diam_ratio>2 AND pos_drift>2\": {np.sum(group_large_ratio & (diag['pos_drift_arcsec'] > 2.0)):,d}")

    # More targeted group flagging: require BOTH diameter increase AND position drift
    group_suspicious_refined = (diag['GROUP_MULT'] > 1) & (
        (diag['pos_drift_arcsec'] > 2.0) |  # position moved
        (diag['diam_ratio'] < 0.5) |         # shrunk significantly (likely real problem)
        ((diag['diam_ratio'] > 2.0) & (diag['pos_drift_arcsec'] > 1.0)) |  # grew AND moved
        ((diag['diam_ratio'] > 3.0) & (diag['DIAM_ORIG'] > 0.5))  # grew 3x for non-tiny galaxy
    )
    log.info(f"Refined suspicious groups: {np.sum(group_suspicious_refined):,d}")

    # Recalculate total with refined group criterion
    large_pos_drift = diag['pos_drift_arcsec'] > 5.0
    extreme_diam = (diag['diam_ratio'] < 0.3) | (diag['diam_ratio'] > 3.0)
    has_largeshift = (diag['ELLIPSEBIT'] & ELLIPSEBIT['LARGESHIFT']) != 0
    has_notractor = (diag['ELLIPSEBIT'] & ELLIPSEBIT['NOTRACTOR']) != 0
    has_skiptractor = (diag['ELLIPSEBIT'] & ELLIPSEBIT['SKIPTRACTOR']) != 0

    group_suspicious_refined = (diag['GROUP_MULT'] > 1) & (
        (diag['pos_drift_arcsec'] > 2.0) |
        (diag['diam_ratio'] < 0.5) |
        ((diag['diam_ratio'] > 2.0) & (diag['pos_drift_arcsec'] > 1.0)) |
        ((diag['diam_ratio'] > 3.0) & (diag['DIAM_ORIG'] > 0.5))
    )

    needs_refit_refined = (
        large_pos_drift |
        extreme_diam |
        has_largeshift |
        #has_notractor |
        #has_skiptractor |
        group_suspicious_refined
    )

    log.info(f"Refined total for refit: {np.sum(needs_refit_refined):,d}/{len(diag):,d}")

    largeshift_no_drift = ((diag['ELLIPSEBIT'] & ELLIPSEBIT['LARGESHIFT']) != 0) & (diag['pos_drift_arcsec'] <= 5.0)

    # Were these "fixed" in a previous version?
    print(f"LARGESHIFT but no drift - median pos_drift: {np.median(diag['pos_drift_arcsec'][largeshift_no_drift]):.2f}")
    print(f"LARGESHIFT but no drift - median diam_ratio: {np.median(diag['diam_ratio'][largeshift_no_drift]):.2f}")
    print(f"LARGESHIFT but no drift - in groups: {np.sum(diag['GROUP_MULT'][largeshift_no_drift] > 1):,d}")

    # Objects with LARGESHIFT that appear stable now
    largeshift_stable = (
        ((diag['ELLIPSEBIT'] & ELLIPSEBIT['LARGESHIFT']) != 0) &
        (diag['pos_drift_arcsec'] <= 5.0) &
        (diag['diam_ratio'] > 0.5) &
        (diag['diam_ratio'] < 2.0)
    )
    log.info(f"LARGESHIFT but stable: {np.sum(largeshift_stable):,d}")

    # Exclude these from refit
    needs_refit_final = needs_refit_refined & ~largeshift_stable
    log.info(f"Final total for refit: {np.sum(needs_refit_final):,d}/{len(diag):,d}")

    is_lvd = diag['IS_LVD']
    log.info(f"LVD sources flagged for refit: {np.sum(needs_refit_refined & is_lvd):,d}/{np.sum(is_lvd):,d}")

    lvd_not_flagged = is_lvd & ~needs_refit_final
    log.info(f"LVD not flagged - median pos_drift: {np.median(diag['pos_drift_arcsec'][lvd_not_flagged]):.2f}")
    log.info(f"LVD not flagged - median diam_ratio: {np.median(diag['diam_ratio'][lvd_not_flagged]):.2f}")
    log.info(f"LVD not flagged - max pos_drift: {np.max(diag['pos_drift_arcsec'][lvd_not_flagged]):.2f}")

    diag['NEEDS_REFIT'] = needs_refit_final

    return diag


def additional_sanity_checks(ell60, ell22):
    """More checks to catch problematic objects."""

    # 1. Objects where flux is zero but D26 is large (Tractor dropped it)
    zero_flux = (ell60['FLUX_R'] == 0) & (ell60['D26'] > 0.5)

    # 2. Objects with valid D26_ERR but suspiciously small D26
    small_but_measured = (ell60['D26_ERR'] > 0) & (ell60['D26'] < 0.2)

    # 3. BA became unphysical
    bad_ba = (ell60['BA'] <= 0) | (ell60['BA'] > 1)

    # 4. Compare to v0.22 measured values (not just init)
    # If v0.22 had a good measurement and v0.60 differs wildly, suspicious
    m60, m22 = match(ell60['OBJNAME'], ell22['OBJNAME'])
    v22_had_measurement = ell22['D26_ERR'][m22] > 0
    diam_changed_from_v22 = np.abs(ell60['D26'][m60] - ell22['D26'][m22]) / np.clip(ell22['D26'][m22], 0.1, None)
    v22_to_v60_drift = v22_had_measurement & (diam_changed_from_v22 > 0.5)

    log.info(f"Zero flux but large D26: {np.sum(zero_flux):,d}")
    log.info(f"Small D26 but measured: {np.sum(small_but_measured):,d}")
    log.info(f"Unphysical BA: {np.sum(bad_ba):,d}")
    log.info(f"Large change from v0.22 measurement: {np.sum(v22_to_v60_drift):,d}")


def prepare_v070_ellipse(ell1, outdir, region, mindiam=0.5):
    """Prepare v0.70 ellipse catalog for v0.80 parent building.

    Reads original v0.10 parent, applies overlay updates, flags objects
    for restoration or removal.

    Parameters
    ----------
    ell1 : Table
        v0.70 ellipse catalog for one region
    outdir : str
        Directory containing parent catalogs
    region : str
        'dr11-south' or 'dr9-north'
    mindiam : float
        Minimum diameter threshold in arcmin

    Returns
    -------
    ell1 : Table
        Modified catalog with RESTORE column and _ORIG columns added,
        small group members removed

    """
    from SGA.SGA import SAMPLE
    from SGA.ellipse import ELLIPSEBIT

    # Read original parent and apply all overlay updates
    parent_orig = Table(fitsio.read(os.path.join(outdir, 'SGA2025-parent-v0.10.fits')))
    for ver in ['v0.30', 'v0.40', 'v0.50', 'v0.60', 'v0.70']:
        overlay_dir = resources.files('SGA').joinpath(f'data/SGA2025/overlays/{ver}')
        ov = load_overlays(overlay_dir)
        I = np.isin(ov.updates['OBJNAME'], parent_orig['OBJNAME'])
        ovup = ov.updates[I]
        apply_updates_inplace(parent_orig, ovup)

    # Match ell1 to parent_orig
    m_ell, m_parent = match(ell1['OBJNAME'], parent_orig['OBJNAME'])
    parent_in_ell = np.isin(ell1['OBJNAME'], parent_orig['OBJNAME'])

    # --- Flag LVD sources that shrunk ---
    is_lvd = np.zeros(len(ell1), dtype=bool)
    is_lvd[m_ell] = (ell1['SAMPLE'][m_ell] & SAMPLE['LVD']) != 0

    diam_ratio = np.ones(len(ell1), dtype=np.float32)
    diam_ratio[m_ell] = ell1['D26'][m_ell] / np.clip(parent_orig['DIAM'][m_parent], 0.01, None)

    shrunk_lvd = is_lvd & (diam_ratio < 0.5)
    log.info(f"LVD sources shrunk to <50% of initial: {np.sum(shrunk_lvd):,d}")

    # --- Flag moment-based diameters that shrunk ---
    d26_ref = np.char.strip(ell1['D26_REF'].astype(str))
    is_mom = (d26_ref == 'mom') | np.char.endswith(d26_ref, '/mom')
    shrunk_mom = parent_in_ell & is_mom & (diam_ratio < 0.5)
    log.info(f"Objects with D26_REF='mom' and shrunk to <50% of initial: {np.sum(shrunk_mom):,d}")

    ## --- Restore NOTRACTOR sources --
    #notractor = parent_in_ell & (ell1['ELLIPSEBIT'] & ELLIPSEBIT['NOTRACTOR'] != 0) & (ell1['ELLIPSEBIT'] & ELLIPSEBIT['FIXGEO'] == 0)

    # --- Flag small group members for removal ---
    remove = _flag_small_for_removal(ell1, mindiam=mindiam)

    # --- Combine restoration flags ---
    refit = shrunk_mom
    #refit = shrunk_mom | notractor
    #refit = shrunk_lvd | shrunk_mom

    # --- Add tracking columns ---
    ell1['REFIT'] = refit
    ell1['RA_ORIG'] = np.zeros(len(ell1), dtype='f8')
    ell1['DEC_ORIG'] = np.zeros(len(ell1), dtype='f8')
    ell1['DIAM_ORIG'] = np.zeros(len(ell1), dtype='f4')
    ell1['DIAM_ORIG_REF'] = np.zeros(len(ell1), dtype='<U14')
    ell1['PA_ORIG'] = np.zeros(len(ell1), dtype='f4')
    ell1['BA_ORIG'] = np.zeros(len(ell1), dtype='f4')

    ell1['RA_ORIG'][m_ell] = parent_orig['RA'][m_parent]
    ell1['DEC_ORIG'][m_ell] = parent_orig['DEC'][m_parent]
    ell1['DIAM_ORIG'][m_ell] = parent_orig['DIAM'][m_parent]
    ell1['DIAM_ORIG_REF'][m_ell] = parent_orig['DIAM_REF'][m_parent]
    ell1['PA_ORIG'][m_ell] = parent_orig['PA'][m_parent]
    ell1['BA_ORIG'][m_ell] = parent_orig['BA'][m_parent]
    try:
        assert(np.all(ell1['RA_ORIG'][refit] != 0.))
    except:
        pdb.set_trace()

    log.info(f'{region}: {np.sum(refit):,d}/{len(ell1):,d} flagged for geometry restoration')
    log.info(f'{region}: Removing {np.sum(remove):,d}/{len(ell1):,d} small group members')

    #from SGA.qa import to_skyviewer_table
    ##check = ell1[remove]
    ##check = ell1[shrunk_lvd]
    #check = ell1[shrunk_mom]
    #check = check[np.argsort(check['D26'])]#[::-1]]
    #check.rename_column('D26', 'DIAM')
    #view = to_skyviewer_table(check)
    #view.write('viewer.fits', overwrite=True)
    #check['OBJNAME', 'GROUP_NAME', 'RA', 'DEC', 'DIAM', 'D26_REF', 'DIAM_ORIG', 'DIAM_ORIG_REF']
    ell1 = ell1[~remove]

    return ell1


def _flag_small_for_removal(ell1, mindiam=0.5, keep_one_survivor=False):
    """Flag small group members for removal.

    Removal criteria:
    - GROUP_MULT=1: remove if (D26+D26_ERR) < mindiam
    - GROUP_MULT>1, not interacting: remove if (D26+D26_ERR) < 20 arcsec
    - GROUP_MULT>1, interacting: remove if much smaller than companions

    Always keep:
    - LVD sources
    - GROUP_PRIMARY

    Parameters
    ----------
    ell1 : Table
        Ellipse catalog
    mindiam : float
        Minimum diameter for singletons (arcmin)
    keep_one_survivor : bool
        If True, keep the largest member if all would be removed

    """
    from SGA.SGA import SAMPLE
    from SGA.ellipse import ELLIPSEBIT

    is_lvd = (ell1['SAMPLE'] & SAMPLE['LVD']) != 0
    is_primary = ell1['GROUP_PRIMARY']

    d26 = np.asarray(ell1['D26'], dtype=np.float32)
    d26_err = np.asarray(ell1['D26_ERR'], dtype=np.float32)
    d26_ul = d26 + d26_err

    mult = ell1['GROUP_MULT']
    is_singleton = mult == 1
    is_in_group = mult > 1

    has_overlap = (ell1['ELLIPSEBIT'] & ELLIPSEBIT['OVERLAP']) != 0
    has_blended = (ell1['ELLIPSEBIT'] & ELLIPSEBIT['BLENDED']) != 0
    is_interacting = has_overlap | has_blended

    # Case 1: Singletons - remove if d26_ul < mindiam
    remove_singleton = is_singleton & (d26_ul < mindiam) & ~is_lvd & ~is_primary

    # Case 2: In group, not interacting - remove if d26_ul < 20 arcsec
    small_thresh = 20. / 60.  # 20 arcsec in arcmin
    remove_group_noninteracting = is_in_group & ~is_interacting & (d26_ul < small_thresh) & ~is_lvd & ~is_primary

    # Case 3: In group, interacting - check diameter ratio to companions
    unique_groups, group_indices = np.unique(ell1['GROUP_NAME'], return_inverse=True)
    n_groups = len(unique_groups)

    # Compute max D26 per group
    max_d26_per_group = np.zeros(n_groups, dtype=np.float32)
    np.maximum.at(max_d26_per_group, group_indices, d26)

    # Compute second-max D26 per group
    is_group_max = d26 == max_d26_per_group[group_indices]
    d26_masked = d26.copy()
    d26_masked[is_group_max] = -np.inf
    second_max_per_group = np.full(n_groups, -np.inf, dtype=np.float32)
    np.maximum.at(second_max_per_group, group_indices, d26_masked)
    second_max_per_group[second_max_per_group == -np.inf] = 0.

    # max_other_d26: for each object, max D26 of other members
    max_other_d26 = np.where(is_group_max,
                              second_max_per_group[group_indices],
                              max_d26_per_group[group_indices])
    max_other_d26[is_singleton] = 0.

    # Remove interacting source if:
    # - D26 < 20 arcsec AND
    # - largest companion > mindiam AND
    # - ratio to largest companion < 0.5
    very_small = d26 < small_thresh
    companion_large = max_other_d26 > mindiam
    ratio_to_companion = d26 / np.clip(max_other_d26, 0.01, None)
    much_smaller = ratio_to_companion < 0.5

    remove_group_interacting = (is_in_group & is_interacting & very_small &
                                 companion_large & much_smaller & ~is_lvd & ~is_primary)

    remove = remove_singleton | remove_group_noninteracting | remove_group_interacting

    # Safety check: ensure at least one member survives per group
    if keep_one_survivor:
        keep = ~remove
        any_kept_per_group = np.zeros(n_groups, dtype=bool)
        np.logical_or.at(any_kept_per_group, group_indices, keep)

        no_survivors = ~any_kept_per_group[group_indices]
        is_largest_in_doomed_group = no_survivors & is_group_max
        remove[is_largest_in_doomed_group] = False

    # Statistics
    log.info(f"Small group member removal:")
    log.info(f"  Singletons (d26_ul < {mindiam}): {np.sum(remove_singleton):,d}")
    log.info(f"  Group, non-interacting (d26_ul < 20\"): {np.sum(remove_group_noninteracting):,d}")
    log.info(f"  Group, interacting shreds: {np.sum(remove_group_interacting):,d}")
    log.info(f"  Total flagged for removal: {np.sum(remove):,d}")

    for m in [1, 2, 3, 4, 5]:
        if m < 5:
            mask = (mult == m) & remove
            log.info(f"    Removing from mult={m}: {np.sum(mask):,d}")
        else:
            mask = (mult >= m) & remove
            log.info(f"    Removing from mult>={m}: {np.sum(mask):,d}")

    return remove


def read_base_ellipse(outdir, base_version, mindiam=0.5):
    """Read the base ellipse catalogs for dr11-south and
    dr9-north. Consolidate duplicates by OBJNAME, combining REGION
    bits (but prefering DR11 if duplicated).

    """
    from SGA.SGA import SAMPLE
    from SGA.coadds import REGIONBITS
    from SGA.ellipse import ELLIPSEMODE, ELLIPSEBIT

    ell = []
    for region in ['dr11-south', 'dr9-north']:
        basefile = os.path.join(outdir, f'SGA2025-beta-{base_version}-{region}.fits')
        ell1 = Table(fitsio.read(basefile))
        log.info(f'Read {len(ell1):,d} rows from {basefile}')

        # In v0.40 there was a bug in SGA_diameter, so
        # recompute. We'll censor below.
        if base_version == 'v0.40':
            from SGA.SGA import SGA_diameter
            diam, diam_err, diam_ref, _ = SGA_diameter(ell1, region)

            I = (ell1['D26_ERR'] != 0.)
            for col, val in zip(['D26', 'D26_ERR', 'D26_REF'], [diam[I], diam_err[I], diam_ref[I]]):
                ell1[col][I] = val

            #import matplotlib.pyplot as plt
            #fig, ax = plt.subplots()
            #for ref in set(ell1['D26_REF'][I]):
            #    J = np.where(ref == ell1['D26_REF'][I])[0]
            #    ax.scatter(ell1['D26'][I][J], diam[I][J], s=1, label=ref)
            #ax.legend()
            #ax.set_xlim(0.2, 18)
            #ax.set_ylim(0.2, 18)
            #ax.plot([0.2, 18], [0.2, 18], color='k')
            #fig.savefig('ioannis/tmp/junk.png')
            # ell1[I][np.where((ell1['D26'][I]) > 14 * (diam[I] < 12))[0]]['OBJNAME', 'RA', 'DEC', 'D26', 'DIAM_INIT']

        elif base_version == 'v0.50':

            d26_ul = ell1['D26'] + ell1['D26_ERR']
            max_diam_per_group = np.zeros(ell1['GROUP_ID'].max() + 1, dtype=np.float32)
            np.maximum.at(max_diam_per_group, ell1['GROUP_ID'], d26_ul)

            I = ((ell1['SAMPLE'] & SAMPLE['LVD'] == 0) &
                (ell1['ELLIPSEBIT'] & ELLIPSEBIT['LARGESHIFT'] == 0) & # still need to check
                #(ell1['D26_ERR'] != 0) &                              # do not include missing
                (max_diam_per_group[ell1['GROUP_ID']] < mindiam))
            groups_to_remove = np.unique(ell1['GROUP_NAME'][I])
            I = np.isin(ell1['GROUP_NAME'], groups_to_remove)

            log.info(f'Removing {np.sum(I):,d}/{len(ell1):,d} {region} galaxies with D(26)<{mindiam:.2f} arcmin')
            ell1 = ell1[~I]

        elif base_version == 'v0.60':
            # Fix a bug in the diameters of SKIPTRACTOR sources before proceeding.
            from SGA.SGA import SGA_diameter
            diam, diam_err, diam_ref, _ = SGA_diameter(ell1, region)
            I = ell1['D26'] != diam
            if np.any(I):
                for col, val in zip(['D26', 'D26_ERR', 'D26_REF'], [diam[I], diam_err[I], diam_ref[I]]):
                    ell1[col][I] = val

            diag = diagnose_drift(ell1, outdir, ref_version='v0.10')
            diag = flag_for_refit(diag, pos_thresh_arcsec=5.0, diam_ratio_lo=0.3, diam_ratio_hi=3.0)
            #bb = diag[diag['NEEDS_REFIT']] ; bb = bb[np.argsort(bb['DIAM_ORIG'])]

            # Add refit flag to ell1 before vstacking
            m_ell, m_diag = match(ell1['SGAID'], diag['SGAID'])
            ell1['REFIT'] = np.zeros(len(ell1), dtype=bool)
            ell1['RA_ORIG'] = np.zeros(len(ell1), dtype='f8')
            ell1['DEC_ORIG'] = np.zeros(len(ell1), dtype='f8')
            ell1['DIAM_ORIG'] = np.zeros(len(ell1), dtype='f4')
            ell1['DIAM_ORIG_REF'] = np.zeros(len(ell1), dtype='<U14')
            ell1['PA_ORIG'] = np.zeros(len(ell1), dtype='f4')
            ell1['BA_ORIG'] = np.zeros(len(ell1), dtype='f4')

            ell1['REFIT'][m_ell] = diag['NEEDS_REFIT'][m_diag]
            ell1['RA_ORIG'][m_ell] = diag['RA_ORIG'][m_diag]
            ell1['DEC_ORIG'][m_ell] = diag['DEC_ORIG'][m_diag]
            ell1['DIAM_ORIG'][m_ell] = diag['DIAM_ORIG'][m_diag]
            ell1['DIAM_ORIG_REF'][m_ell] = diag['DIAM_ORIG_REF'][m_diag]
            ell1['PA_ORIG'][m_ell] = diag['PA_ORIG'][m_diag]
            ell1['BA_ORIG'][m_ell] = diag['BA_ORIG'][m_diag]

            log.info(f'{region}: {np.sum(ell1["REFIT"]):,d}/{len(ell1):,d} flagged for refit')

        elif base_version == 'v0.70':
            ell1 = prepare_v070_ellipse(ell1, outdir, region, mindiam=mindiam)

        ell.append(ell1)
    ell = vstack(ell)

    # Consolidate duplicate OBJNAMEs (combine REGION bits), prefer more BANDS; tie→dr11-south
    if len(np.unique(ell['OBJNAME'])) != len(ell):
        #log.info('Consolidating duplicate OBJNAME entries')

        # Precompute band counts and DR11 bit for preference
        def _nbands_val(v):
            try:
                s = (v or '').strip()
            except Exception:
                s = ''
            return len(set(s)) if s else 0

        nbands   = np.array([_nbands_val(b) for b in ell['BANDS']], dtype=int)
        dr11_bit = REGIONBITS['dr11-south']

        def _prefer_idx(rows):
            # First prefer unflagged (REFIT=False)
            unflagged = [i for i in rows if not ell['REFIT'][i]]
            if unflagged:
                rows = np.array(unflagged)

            best = np.max(nbands[rows])
            cand = rows[nbands[rows] == best]
            # tie-breaker: prefer a dr11-south row
            for i in cand:
                if (int(ell['REGION'][i]) & dr11_bit) != 0:
                    return i
            return cand[0]

        names = np.asarray(ell['OBJNAME'])
        uniq, inv, counts = np.unique(names, return_inverse=True, return_counts=True)

        # Keep all singletons immediately
        row_is_singleton = (counts[inv] == 1)
        keep_single = np.flatnonzero(row_is_singleton)

        # Process only duplicate groups
        dup_uinds = np.flatnonzero(counts > 1)
        keep_dup = []
        new_reg_vals = []

        for ui in dup_uinds:
            rows = np.flatnonzero(inv == ui)
            keep = _prefer_idx(rows)
            keep_dup.append(keep)
            new_reg_vals.append(int(np.bitwise_or.reduce(ell['REGION'][rows])))

        keep_dup = np.array(keep_dup, dtype=int)

        # Build final index list (preserve original order)
        keep_all = np.concatenate([keep_single, keep_dup])
        keep_all.sort()

        ell = ell[keep_all]  # subset

        # Update REGION for the kept rows that came from duplicate groups
        # Map original kept index -> new REGION, then assign in the subset
        orig_to_newreg = dict(zip(keep_dup.tolist(), new_reg_vals))
        # find, inside the subset, which rows came from dup keeps
        subset_orig_idx = keep_all  # original indices before subsetting
        update_pos = [i for i, orig in enumerate(subset_orig_idx) if orig in orig_to_newreg]
        if update_pos:
            reg = np.array(ell['REGION'], copy=True)
            for i in update_pos:
                reg[i] = orig_to_newreg[subset_orig_idx[i]]
            ell['REGION'] = reg

        log.info(f'Combined catalog contains {len(ell):,d} unique objects')

    # Project ellipse to parent base model
    ell_base = Table()
    ell_base['SGAID'] = ell['SGAID'].astype(np.int64)
    ell_base['REGION'] = ell['REGION'].astype(np.int16)
    ell_base['OBJNAME'] = ell['OBJNAME'].astype('U30')
    ell_base['PGC'] = ell['PGC'].astype(np.int32)
    ell_base['RA'] = ell['RA'].astype(np.float64)
    ell_base['DEC'] = ell['DEC'].astype(np.float64)
    ell_base['DIAM'] = ell['D26'].astype(np.float32)
    ell_base['DIAM_ERR'] = ell['D26_ERR'].astype(np.float32)
    ell_base['BA'] = ell['BA'].astype(np.float32)
    ell_base['PA'] = (ell['PA'] % 180.).astype(np.float32)
    ell_base['MAG'] = ell['MAG_INIT'].astype(np.float32)
    ell_base['DIAM_REF'] = ell['D26_REF'].astype('U14')

    I = ell['D26_ERR'] != 0. # re-measured (ellipse, not "missing") diameters
    if np.any(I):
        ell_base['DIAM_REF'][I] = np.char.add(f'{base_version}/', ell_base['DIAM_REF'][I])

    # reset SAMPLE except for LVD
    ell_base['SAMPLE'] = np.zeros(len(ell), np.int32) # ell['SAMPLE'].astype(np.int32)
    ell_base['SAMPLE'][ell['SAMPLE'] & SAMPLE['LVD'] != 0] |= SAMPLE['LVD']

    # Not all ellipse entries are present or reliable; read the
    # base_parent catalog so we can revert as appropriate.
    parent_basebasefile = os.path.join(outdir, f'SGA2025-beta-parent-{base_version}.fits')
    parent_base = Table(fitsio.read(parent_basebasefile))
    log.info(f'Read {len(parent_base):,d} rows from {parent_basebasefile}')

    # in v0.50 and higher the ellipse catalogs were reliable enough
    # that we could start to trim "small" galaxies from the sample
    # (see above), so only do the assert for earlier versions
    if float(base_version[1:]) < 0.5:
        assert(len(ell) == len(parent_base))
    assert(len(ell) == len(ell_base))

    m_ell, m_parent = match(ell['OBJNAME'], parent_base['OBJNAME'])
    ell = ell[m_ell]
    ell_base = ell_base[m_ell]
    parent_base = parent_base[m_parent]

    assert(np.all(parent_base['SGAID'] == ell['SGAID']))
    assert(np.all(parent_base['SGAID'] == ell_base['SGAID']))

    return ell, ell_base, parent_base


def build_parent(mp=1, mindiam=0.5, base_version='v0.70', overwrite=False):

    """Build a new parent catalog starting from `base_version` ellipse
    catalog, apply versioned overlays (adds/updates/drops/flags),
    re-derive bits, build groups, and write a single FITS output.

    Notes
    -----
    - Assumes overlay helpers are already available in scope:
      load_overlays, apply_drops_inplace, apply_adds_inplace,
      apply_updates_inplace, apply_flags_inplace.
    - Uses nocuts v0.22 only to restore properties for rows added via overlays.
    - SGAID stability: carried from ellipse (or ROW_PARENT) when present;
      new rows get monotonically increasing SGAID.

    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from desiutil.dust import SFDMap
    from SGA.SGA import SGA_version, SAMPLE, SGA_diameter
    from SGA.coadds import REGIONBITS
    from SGA.groups import build_group_catalog, make_singleton_group, set_overlap_bit
    from SGA.ellipse import ELLIPSEMODE, FITMODE, ELLIPSEBIT
    from SGA.sky import find_in_mclouds, find_in_gclpne

    # Paths & versions
    parent_version = SGA_version(parent=True)      # -> 'v0.30' (target)
    nocuts_version = SGA_version(nocuts=True)      # -> 'v0.22'
    outdir = os.path.join(sga_dir(), 'sample')
    parentdir = os.path.join(sga_dir(), 'parent')
    overlay_dir = resources.files('SGA').joinpath(f'data/SGA2025/overlays/{parent_version}') # e.g., .../overlays/v0.30
    outfile = os.path.join(outdir, f'SGA2025-beta-parent-{parent_version}.fits')
    kdoutfile = os.path.join(outdir, f'SGA2025-beta-parent-{parent_version}.kd.fits')

    if os.path.isfile(outfile) and not overwrite:
        log.info(f'Parent catalog {outfile} exists; use --overwrite')
        return

    # Read the base ellipse catalogs for dr11-south and
    # dr9-north.
    ell, ell_base, parent_base = read_base_ellipse(outdir, base_version, mindiam=mindiam)
    assert(np.all(np.isfinite(ell['D26'])))

    if base_version == 'v0.22':
        # Restore all objects with large shifts caused by erroneous
        # Tractor models and also the geometry measured for all LVD
        # sources.

        # After much inspection, I decided to also restore the
        # geometry for every object in v0.22 where the diameter,
        # position angle, or ellipticity changed by more than 20% from
        # its initial / parent values.
        d_old = parent_base['DIAM'].value
        d_new = ell_base['DIAM'].value
        ba_old = parent_base['BA'].value
        ba_new = ell_base['BA'].value
        pa_old = parent_base['PA'].value  # degrees, [0,180)
        pa_new = ell_base['PA'].value

        # 1) keep existing special cases
        I1 = ((ell['ELLIPSEBIT'] & ELLIPSEBIT['LARGESHIFT'] != 0) |
              (ell['SAMPLE'] & SAMPLE['LVD'] != 0) |
              (ell['GROUP_MULT'] > 1))

        # 2) relative change > 20% for DIAM and BA (guard old<=0)
        I2 = (d_old > 0) & (np.abs(d_new - d_old) / d_old > 0.20)

        # BA can be near zero; use relative when possible, else absolute > 0.20
        I3_rel = (ba_old > 0) & (np.abs(ba_new - ba_old) / ba_old > 0.20)
        I3_abs = (ba_old <= 0) & (np.abs(ba_new - ba_old) > 0.20)
        I3 = I3_rel | I3_abs

        # 3) PA: use wrapped angular difference in [0,90], then compare to 20% of 180° (=36°)
        # (i.e., “20% change” interpreted on the 180° periodicity)
        dpa = np.abs(((pa_new - pa_old + 90.0) % 180.0) - 90.0)  # wrapped |ΔPA| in degrees
        I4 = dpa > (0.20 * 180.0)  # 36 degrees

        I = I1 | I2 | I3 | I4
        #ell_base['OBJNAME', 'RA', 'DEC', 'DIAM', 'BA', 'PA'][I][:10]
        #parent_base['OBJNAME', 'RA', 'DEC', 'DIAM', 'BA', 'PA'][I][:10]
        if np.any(I):
            log.info(f'Restoring positions and ellipse geometry for {np.sum(I):,d} objects.')
            ell_base['DIAM_ERR'][I] = 0.
            for col in ['RA', 'DEC', 'DIAM', 'PA', 'BA', 'DIAM_REF']:
                ell_base[col][I] = parent_base[col][I]

    elif base_version == 'v0.30':
        I = ((ell['SAMPLE'] & SAMPLE['LVD']) == 0) & (ell['D26_ERR'] != 0.) & ((ell['D26']+ell['D26_ERR']) < mindiam)
        log.info(f'Removing {np.sum(I):,d}/{len(ell):,d} galaxies with D(26)<{mindiam:.2f} arcmin')
        base = ell_base[~I]

    elif base_version == 'v0.40':
        # buggy diameters in v0.30, so re-compute and then restore the
        # objects that are actually larger than 30 arcsec
        vell, vell_base, vparent_base = read_base_ellipse(outdir, 'v0.30')

        I = ((vell['SAMPLE'] & SAMPLE['LVD']) == 0) & (vell['D26_ERR'] != 0.) & ((vell['D26']+vell['D26_ERR']) < mindiam)
        vell = vell[I]
        vell_base = vell_base[I]
        vparent_base = vparent_base[I]

        diam, diam_err, diam_ref, _ = SGA_diameter(vell, 'dr11-south')
        I = ((diam+diam_err) > mindiam) & (vell['GROUP_MULT'] == 1)
        log.info(f'Restoring {np.sum(I):,d} galaxies incorrectly dropped in v0.30.')

        ell = vstack((ell, vell[I]))
        ell_base = vstack((ell_base, vell_base[I]))
        parent_base = vstack((parent_base, vparent_base[I]))

        # still not ready to trust the new diameters in groups; IC
        # 4721A is an example object where we do not want the larger,
        # newer diameter
        I1 = ((ell['ELLIPSEBIT'] & ELLIPSEBIT['LARGESHIFT'] != 0) |
              (ell['SAMPLE'] & SAMPLE['LVD'] != 0) |
              (ell['GROUP_MULT'] > 1))

        d_old = parent_base['DIAM'].value
        d_new = ell_base['DIAM'].value
        I2 = (d_old > 0) & (np.abs(d_new - d_old) / d_old > 0.20)

        I = I1 | I2
        if np.any(I):
            log.info(f'Restoring original ellipse geometry for {np.sum(I):,d} objects.')
            ell_base['DIAM_ERR'][I] = 0.
            for col in ['RA', 'DEC', 'DIAM', 'PA', 'BA', 'DIAM_REF']:
                ell_base[col][I] = parent_base[col][I]

        base = ell_base

    elif base_version == 'v0.50':
        ## still not ready to trust the new diameters in groups with
        ## the overlap bit set or near bright stars; IC 4721A is an
        ## example object where we do not want the larger, newer
        ## diameter
        #I = ((ell['SAMPLE'] & SAMPLE['NEARSTAR'] != 0) |
        #    (ell['ELLIPSEBIT'] & ELLIPSEBIT['OVERLAP'] != 0))
        #
        #log.info(f'Restoring original ellipse geometry for {np.sum(I):,d} objects.')
        #ell_base['DIAM_ERR'][I] = 0.
        #for col in ['RA', 'DEC', 'DIAM', 'PA', 'BA', 'DIAM_REF']:
        #    ell_base[col][I] = parent_base[col][I]
        base = ell_base

    elif base_version == 'v0.60':
        I = ell['REFIT'].astype(bool)
        if np.any(I):
            log.info(f'Restoring initial geometry for {np.sum(I):,d} objects for refit')
            ell_base['DIAM_ERR'][I] = 0.
            for col, init_col in [('RA', 'RA_ORIG'), ('DEC', 'DEC_ORIG'),
                                  ('DIAM', 'DIAM_ORIG'), ('DIAM_REF', 'DIAM_ORIG_REF'),
                                  ('PA', 'PA_ORIG'), ('BA', 'BA_ORIG')]:
                if ell[init_col][I] == 0.:
                    pdb.set_trace()
                ell_base[col][I] = ell[init_col][I]

            out = ell['SGAID', 'OBJNAME', 'RA_ORIG', 'DEC_ORIG', 'REGION', 'SAMPLE', 'DIAM_ORIG', 'PA_ORIG', 'BA_ORIG'][I]
            out = out[np.argsort(out['DIAM_ORIG'])]
            out.write(os.path.join(outdir, 'SGA2025-v0.70-refit.fits'), overwrite=True)
        base = ell_base
    elif base_version == 'v0.70':
        I = ell['REFIT'].astype(bool)
        if np.any(I):
            log.info(f'Restoring initial geometry for {np.sum(I):,d} objects for refit')
            ell_base['DIAM_ERR'][I] = 0.
            for col, init_col in [('RA', 'RA_ORIG'), ('DEC', 'DEC_ORIG'),
                                  ('DIAM', 'DIAM_ORIG'), ('DIAM_REF', 'DIAM_ORIG_REF'),
                                  ('PA', 'PA_ORIG'), ('BA', 'BA_ORIG')]:
                ell_base[col][I] = ell[init_col][I]

            out = ell['SGAID', 'OBJNAME', 'RA_ORIG', 'DEC_ORIG', 'REGION', 'SAMPLE', 'DIAM_ORIG', 'PA_ORIG', 'BA_ORIG'][I]
            out = out[np.argsort(out['DIAM_ORIG'])]
            out.write(os.path.join(outdir, 'SGA2025-v0.80-refit.fits'), overwrite=True)
        base = ell_base
    else:
        base = ell_base

    log.info(f'Final base catalog contains {len(base):,d} objects.')

    # Apply overlays (drops, adds [with nocuts restore], updates, flags)
    ov = load_overlays(overlay_dir)
    nocuts_file = os.path.join(parentdir, f'SGA2025-parent-nocuts-{nocuts_version}.fits')
    nocuts = Table(fitsio.read(nocuts_file, columns=['OBJNAME', 'PGC', 'ROW_PARENT']))
    log.info(f'Read {len(nocuts):,d} rows from {nocuts_file}')

    base = apply_drops(base, ov.drops, REGIONBITS)
    base = apply_adds(base, ov.adds, REGIONBITS, nocuts)

    miss = ~np.isin(ov.updates['OBJNAME'].value, base['OBJNAME'])
    if np.any(miss):
        log.critical("The following objects in the updates.csv file are missing")
        print(ov.updates['OBJNAME'][miss])
        raise ValueError()
    apply_updates_inplace(base, ov.updates)

    try:
        if len(np.unique(base['SGAID'])) != len(base):
            raise ValueError('Non-unique SGAID in final parent')
        if not np.all(base['DIAM'] > 0.):
            raise ValueError('Non-positive DIAM in final parent')
        if not np.all((base['BA'] > 0.) & (base['BA'] <= 1.)):
            raise ValueError('BA out of range')
        if not np.all((base['PA'] >= 0.) & (base['PA'] < 180.)):
            raise ValueError('PA out of range')
    except:
        base[(base['BA'] <= 0.) | (base['BA'] > 1.)]['OBJNAME', 'RA', 'DEC', 'DIAM', 'BA', 'PA']

    # Check for sources within 3.6 arcsec of each other
    coords = SkyCoord(base['RA'] * u.deg, base['DEC'] * u.deg)
    idx1, idx2, sep, _ = coords.search_around_sky(coords, 3.6 * u.arcsec)

    # Remove self-matches
    not_self = idx1 != idx2
    if np.any(not_self):
        # Get unique pairs (avoid counting i,j and j,i twice)
        pairs = np.array(sorted(set(tuple(sorted((i, j))) for i, j in zip(idx1[not_self], idx2[not_self]))))
        raise ValueError(f"Found {len(pairs)} source pairs within 3.6 arcsec:\n"
                         f"{base['OBJNAME', 'RA', 'DEC'][pairs[:10].flatten()]}")

    # re-add the Gaia masking bits
    add_gaia_masking(base)

    # Initialize the output data model
    grp = base['SGAID', 'REGION', 'OBJNAME', 'PGC']
    grp['SAMPLE'] = np.zeros(len(grp), np.int32) # zero out
    grp['ELLIPSEMODE'] = np.zeros(len(grp), np.int32)
    grp['FITMODE'] = np.zeros(len(grp), np.int32)
    grp['RA'] = base['RA'].astype('f8')
    grp['DEC'] = base['DEC'].astype('f8')
    grp['DIAM'] = base['DIAM'].astype('f4')
    grp['BA'] = base['BA'].astype('f4')
    grp['PA'] = base['PA'].astype('f4')
    grp['MAG'] = base['MAG'].astype('f4')
    grp['DIAM_REF'] = base['DIAM_REF']

    SFD = SFDMap(scaling=1.0)
    grp['EBV'] = SFD.ebv(base['RA'].astype(float), base['DEC'].astype(float)).astype('f4')

    # Populate the SAMPLE bits.
    grp['SAMPLE'][base['SAMPLE'] & SAMPLE['LVD'] != 0] |= SAMPLE['LVD']
    grp['SAMPLE'][base['STARFDIST'] < 1.2] |= SAMPLE['NEARSTAR']
    grp['SAMPLE'][base['STARFDIST'] < 0.5] |= SAMPLE['INSTAR']

    in_LMC = find_in_mclouds(grp, mcloud='LMC')
    in_SMC = find_in_mclouds(grp, mcloud='SMC')
    in_gclpne = find_in_gclpne(grp)
    grp['SAMPLE'][in_LMC | in_SMC] |= SAMPLE['MCLOUDS']
    grp['SAMPLE'][in_gclpne] |= SAMPLE['GCLPNE']

    log.info(f'Applying ELLIPSEMODE flags to {len(ov.flags):,d} objects.')
    apply_flags_inplace(grp, ov.flags, ELLIPSEMODE)

    # MCLOUDS/GCLPNE/NEARSTAR/INSTAR all imply NORADWEIGHT
    I = (grp['SAMPLE'] & (SAMPLE['MCLOUDS'] | SAMPLE['GCLPNE'] | SAMPLE['NEARSTAR'] | SAMPLE['INSTAR'])) != 0
    grp['ELLIPSEMODE'][I] |= ELLIPSEMODE['NORADWEIGHT']

    # populate FITMODE, which is used by legacypipe
    grp['FITMODE'][grp['ELLIPSEMODE'] & ELLIPSEMODE['FIXGEO'] != 0] |= FITMODE['FIXGEO']
    grp['FITMODE'][grp['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED'] != 0] |= FITMODE['RESOLVED']
    assert(np.all(np.isfinite(grp['DIAM'])))

    # Sort by diameter descending and then build the group catalog
    srt = np.argsort(grp['DIAM'])[::-1]
    grp = grp[srt]

    special = ((grp['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED']) != 0) | \
              ((grp['ELLIPSEMODE'] & ELLIPSEMODE['FORCEPSF']) != 0)
    out1 = make_singleton_group(grp[special], group_id_start=0)
    gid_start = int(np.max(out1['GROUP_ID'])) + 1
    out2 = build_group_catalog(grp[~special], group_id_start=gid_start, mp=mp)
    out = vstack((out1, out2))

    # Assign SGAGROUP name and check duplicates among primaries
    try:
        groupname = np.char.add('SGA2025_', out['GROUP_NAME'])
        out.add_column(groupname, name='SGAGROUP', index=1)
        prim = out['GROUP_PRIMARY']
        gg, cc = np.unique(out['SGAGROUP'][prim], return_counts=True)
        if np.any(cc > 1):
            log.critical('Duplicate group names among primaries detected.')
            raise ValueError('Duplicate SGAGROUP among primaries')
    except:
        pdb.set_trace()

    # Harmonize REGION bits within groups (keep only bits common to
    # all members; drop groups with none).
    unique_groups, group_indices = np.unique(out['GROUP_ID'], return_inverse=True)
    n_groups = len(unique_groups)

    # Compute bitwise AND of REGION for each group; start with all bits set
    region_and_per_group = np.full(n_groups, REGIONBITS['dr11-south'] | REGIONBITS['dr9-north'], dtype=np.int16)
    np.bitwise_and.at(region_and_per_group, group_indices, out['REGION'])

    # Get the allowed bits for each row
    allowed = region_and_per_group[group_indices]

    # Groups to drop (no common bits AND mult > 1)
    group_mult = out['GROUP_MULT']
    drop_mask = (allowed == 0) & (group_mult > 1)

    # Groups to strip (some bits removed but group kept)
    new_region = out['REGION'] & allowed
    strip_mask = (new_region != out['REGION']) & (allowed != 0) & (group_mult > 1)

    # Apply changes
    if np.any(strip_mask):
        out['REGION'] = new_region
        strip_groups = np.unique(out['GROUP_NAME'][strip_mask])
        log.info(f"Stripped REGION bits (kept groups) for {len(strip_groups):,d} groups.")

    if np.any(drop_mask):
        drop_ids = np.unique(out['GROUP_ID'][drop_mask])
        log.info(f"Dropping {len(drop_ids):,d} groups with no common REGION bit ({np.sum(drop_mask):,d} members).")
        out = out[~drop_mask]

    # OVERLAP bit
    set_overlap_bit(out, SAMPLE)

    # Final sanity: unique SGAID; DIAM>0; 0<BA≤1; PA∈[0,180)
    if len(np.unique(out['SGAID'])) != len(out):
        raise ValueError('Non-unique SGAID in final parent')
    if not np.all(out['DIAM'] > 0.):
        raise ValueError('Non-positive DIAM in final parent')
    if not np.all((out['BA'] > 0.) & (out['BA'] <= 1.)):
        raise ValueError('BA out of range')
    if not np.all((out['PA'] >= 0.) & (out['PA'] < 180.)):
        raise ValueError('PA out of range')

    # Write
    log.info(f'Writing {len(out):,d} objects to {outfile}')
    out.meta['EXTNAME'] = 'PARENT'
    out.write(outfile, overwrite=True)

    cmd = f'startree -i {outfile} -o {kdoutfile} -T -P -k -n stars'
    log.info(cmd)
    _ = os.system(cmd)
