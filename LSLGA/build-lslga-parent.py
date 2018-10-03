
# coding: utf-8

# # Build the LSLGA parent sample
# 
# The purpose of this notebook is to build the parent sample for the Legacy Surveys Galaxy Atlas (LSGA).  The final output files are:
#   * large-galaxies-parent.fits -- large galaxies in the Legacy Surveys footprint
#   * large-galaxies-parent-dr6-dr7.fits -- large galaxies in the DR6+DR7 footprint
# 
# Our starting catalog is the file hyperleda-d25min10.txt, which contains 2,118,186 objects and is the raw output of querying the [Hyperleda database](http://leda.univ-lyon1.fr/fullsql.html) (on 2018 May 13) for all objects with a D(25) isophotal diameter greater than 10 arcsec using the following SQL query:
# 
# ```SQL
# SELECT
#   pgc, objname, objtype, al2000, de2000, type, bar, ring,
#   multiple, compactness, t, logd25, logr25, pa, bt, it,
#   kt, v, modbest
# WHERE
#   logd25 > 0.2218487 and (objtype='G' or objtype='M' or objtype='M2' or 
#                           objtype='M3' or objtype='MG' or objtype='MC')
# ORDER BY
#   al2000
# ```
# 
# **ToDo**
# 
# 1. Build unWISE and GALEX mosaics.
# 2. Filter and sort the sample; try to remove spurious sources.
# 3. Include additional metadata in the webpage.

# ### Imports and other preliminaries.

# In[15]:


import os, sys
import time
from contextlib import redirect_stdout
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


# In[16]:


import fitsio
import astropy.table
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


# In[17]:


import multiprocessing
nproc = multiprocessing.cpu_count() // 2


# In[18]:


sns.set(style='ticks', font_scale=1.5, palette='Set2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


PIXSCALE = 0.262
mindiameter = 10 / 60  # [arcmin]
maxdiameter = 1e4    # [arcmin]
linking_length = 0.75 # [arcmin]
print('Linking length = {:.2f} arcmin = {:.1f} pixels.'.format(linking_length, linking_length * 60 / PIXSCALE))


# In[20]:


drsuffix = 'dr6-dr7'
LSLGAdir = os.getenv('LSLGA_DIR')
parentfile = os.path.join(LSLGAdir, 'sample', 'large-galaxies-parent.fits')
groupcatfile = os.path.join(LSLGAdir, 'sample', 'large-galaxies-parent-groupcat.fits')
groupsamplefile = os.path.join(LSLGAdir, 'sample', 'large-galaxies-groupcat-{}.fits'.format(drsuffix))
samplefile = os.path.join(LSLGAdir, 'sample', 'large-galaxies-{}.fits'.format(drsuffix))


# In[21]:


viewerurl = 'http://legacysurvey.org/viewer'
cutouturl = 'http://legacysurvey.org/viewer-dev/jpeg-cutout'


# In[22]:


htmlfile = os.path.join(LSLGAdir, 'index.html')
htmlfile_reject = os.path.join(LSLGAdir, 'index-reject.html')


# #### Some choices.

# In[32]:


rebuild_parent = True
rebuild_groupcat = True
rebuild_dr_sample = True


# ### Define some convenience QA functions.

# In[26]:


def qa_radec_dr(parent, sample):
    idr5 = sample['dr'] == 'dr5'
    idr6 = sample['dr'] == 'dr6'
    idr7 = sample['dr'] == 'dr7'

    fig, ax = plt.subplots()
    ax.scatter(parent['ra'], parent['dec'], alpha=0.5, s=5, label='Parent Catalog')
    ax.scatter(sample['ra'][idr5], sample['dec'][idr5], s=10, label='In DR5')
    ax.scatter(sample['ra'][idr6], sample['dec'][idr6], s=10, label='In DR6')
    #ax.scatter(sample['ra'][idr7], sample['dec'][idr7], s=10, label='In DR7')
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.legend(loc='upper left', fontsize=10, frameon=True)#, ncol=3)


# ### Define the parent sample as the set of Hyperleda objects in the LS/DESI footprint.
# 
# We also apply minimum and maximum angular diameter cuts here.
# 
# We need to make sure the bright-star masking doesn't remove objects like [IC 2204](http://legacysurvey.org/viewer?ra=115.3331&dec=34.2240&zoom=12&layer=mzls+bass-dr6), a beautiful r=13.4 disk galaxy at a redshift of z=0.0155 with an angular diameter of approximately 1.1 arcmin.

# In[28]:


def read_hyperleda():
    """Read the Hyperleda catalog.
    
    """
    hyperledafile = os.path.join(LSLGAdir, 'sample', 'hyperleda-d25min10-18may13.fits')
    allwisefile = hyperledafile.replace('.fits', '-allwise.fits')

    leda = astropy.table.Table(fitsio.read(hyperledafile, ext=1))
    leda.add_column(astropy.table.Column(name='groupid', dtype='i4', length=len(leda)))
    print('Read {} objects from {}'.format(len(leda), hyperledafile), flush=True)

    allwise = astropy.table.Table(fitsio.read(allwisefile, ext=1, lower=True))
    print('Read {} objects from {}'.format(len(allwise), allwisefile), flush=True)

    # Merge the tables
    allwise.rename_column('ra', 'wise_ra')
    allwise.rename_column('dec', 'wise_dec')
    
    leda = astropy.table.hstack( (leda, allwise) )
    leda['inwise'] = (np.array(['NULL' not in dd for dd in allwise['designation']]) * 
                      np.isfinite(allwise['w1sigm']) * np.isfinite(allwise['w2sigm']) )
    
    # Require a magnitude estimate.
    magcut = np.isfinite(leda['mag'])
    leda = leda[magcut]
    #print('Removing {} objects with no magnitude estimate.'.format(np.sum(~magcut)))
    
    #print('  Identified {} objects with WISE photometry.'.format(np.sum(leda['inwise'])))
    
    return leda


# In[29]:


def read_tycho(magcut=12):
    """Read the Tycho 2 catalog.
    
    """
    tycho2 = os.path.join(LSLGAdir, 'sample', 'tycho2.kd.fits')
    tycho = astropy.table.Table(fitsio.read(tycho2, ext=1, lower=True))
    tycho = tycho[np.logical_and(tycho['isgalaxy'] == 0, tycho['mag_bt'] <= magcut)]
    print('Read {} Tycho-2 stars with B<{:.1f}.'.format(len(tycho), magcut), flush=True)
    
    # Radius of influence; see eq. 9 of https://arxiv.org/pdf/1203.6594.pdf
    tycho['radius'] = (0.0802*(tycho['mag_bt'])**2 - 1.860*tycho['mag_bt'] + 11.625) / 60 # [degree]
    
    return tycho    


# In[33]:


def build_parent(nside=128):
    """Identify the galaxies in the nominal LS/DESI footprint."""
    import desimodel.io
    import desimodel.footprint
    from astrometry.libkd.spherematch import tree_build_radec, tree_search_radec
    
    leda = read_hyperleda()
    
    tiles = desimodel.io.load_tiles(onlydesi=True)
    indesi = desimodel.footprint.is_point_in_desi(tiles, ma.getdata(leda['ra']), 
                                                  ma.getdata(leda['dec']))
    print('  Removing {} objects outside the DESI footprint.'.format(np.sum(~indesi)), flush=True)

    diamcut = (leda['d25'] >= mindiameter) * (leda['d25'] <= maxdiameter)
    print('  Removing {} objects with D(25) < {:.3f} and D(25) > {:.3f} arcmin.'.format(
        np.sum(~diamcut), mindiameter, maxdiameter), flush=True)

    # Reject objects classified as "g"
    # objnotg = np.hstack([np.char.strip(obj) != 'g' for obj in leda['objtype']])
    # print('  Removing {} objects with objtype == g'.format(np.sum(~objnotg)), flush=True)
    
    keep = np.where( indesi * diamcut )[0]
    parent = leda[keep]
    print('The parent sample before star-masking has {} objects.'.format(len(parent)), flush=True)
    print()
    
    # Next, read the Tycho2 catalog and build a KD tree to flag objects near bright stars.
    #print('Flagging galaxies near bright stars.')
    tycho = read_tycho()
    kdparent = tree_build_radec(parent['ra'], parent['dec'])

    nearstar = np.zeros( len(parent), dtype=bool)
    for star in tycho:
        I = tree_search_radec(kdparent, star['ra'], star['dec'], star['radius'])
        if len(I) > 0:
            nearstar[I] = True
    print('Found {} galaxies near a Tycho-2 star.'.format(np.sum(nearstar)), flush=True)

    # Write out everything (do not reject) but also write out the sample near 
    # bright stars for further analysis.
    badparent = parent[nearstar]
    #goodparent = parent[~nearstar]
    goodparent = parent # this line, not the one above

    print('Writing {} objects to {}'.format(len(goodparent), parentfile), flush=True)
    goodparent.write(parentfile, overwrite=True)    

    badparentfile = parentfile.replace('.fits', '-nearstars.fits')
    print('Writing {} objects to {}'.format(len(badparent), badparentfile), flush=True)
    badparent.write(badparentfile, overwrite=True)    
    print()
    
    return parent, leda


# In[34]:


def read_parent(parentfile, leda=None):
    """Read the previously created parent catalog.
    
    """
    parent = astropy.table.Table(fitsio.read(parentfile, ext=1))
    print('Read {} objects from {}'.format(len(parent), parentfile), flush=True)
    
    if leda is None:
        leda = read_hyperleda()
    
    return parent, leda


# In[35]:


if rebuild_parent:
    parentlogfile = os.path.join(LSLGAdir, 'sample', 'build-parent.log'.format(drsuffix))
    print('Building the parent sample.')
    print('Logging to {}'.format(parentlogfile))
    t0 = time.time()
    with open(parentlogfile, 'w') as log:
        with redirect_stdout(log):
            parent, leda = build_parent()
            print('Total time = {:.3f} min.'.format( (time.time() - t0) / 60 ), flush=True)
    with open(parentlogfile, 'r') as log:
        print(log.read())    
else:
    parent, leda = read_parent(parentfile)


# Some sanity QA.

# In[38]:


from LSLGA.qa import qa_binned_radec


# In[36]:


def qa_mag_d25(cat, supercat):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hexbin(supercat['mag'], np.log10(supercat['d25']), extent=(0, 25, -1, 3),
              mincnt=1, cmap='viridis')
    ax.scatter(cat['mag'], np.log10(cat['d25']), s=5)
    #ax.hexbin(cat['mag'], np.log10(cat['d25']), extent=(5, 23, -1, 2),
    #          mincnt=1, cmap='viridis')
    ax.axhline(y=np.log10(10 / 60), ls='-', lw=2, color='k', alpha=0.8)
    #ax.axhline(y=np.log10(mindiameter), ls='--', lw=2, color='red', alpha=0.8)
    #ax.axhline(y=np.log10(maxdiameter), ls='--', lw=2, color='red', alpha=0.8)
    ax.set_xlabel('B mag')
    ax.set_ylabel(r'$\log_{10}\, D_{25}$ (arcmin)')


# In[37]:


qa_mag_d25(parent, leda)


# In[40]:


#qa_binned_radec(parent)


# ### Construct a group catalog using a friends-of-friends algorithm.

# In[45]:


def read_groupcat(groupcatfile, parentfile, leda=None):
    """Read the previously created group catalog.
    
    """
    groupcat = astropy.table.Table(fitsio.read(groupcatfile, ext=1))
    print('Read {} groups from {}'.format(len(groupcat), groupcatfile), flush=True)

    parent, leda = read_parent(parentfile, leda=leda)    
    
    return groupcat, parent


# In[46]:


rebuild_groupcat=False


# In[47]:


if rebuild_groupcat:
    from LSLGA.groups import build_groupcat_sky
    groupcat, parent = build_groupcat_sky(parent, linking_length=linking_length, 
                                          verbose=True, groupcatfile=groupcatfile,
                                          parentfile=parentfile)
else:
    groupcat, _ = read_groupcat(groupcatfile, parentfile, leda=leda)


# ### Identify the set of groups in the DR6/DR7 footprint.

# In[48]:


def group_diameter(onegroup):
    """Define the diameter or angular extent of the group."""
    return 1.2 * np.max( (3 * onegroup['width'], 3 * onegroup['d25max']) ) # [arcmin]


# In[25]:


def init_survey(dr='dr7'):
    """
    rsync -auvzP cori:"/global/project/projectdirs/cosmo/work/legacysurvey/dr5/survey-ccds-dr5-patched.kd.fits" $LSLGA_DIR/sample/dr5/
    rsync -auvzP cori:"/global/cscratch1/sd/dstn/dr6plus/survey-ccds-dr6plus.kd.fits" $LSLGA_DIR/sample/dr6/
    rsync -auvzP cori:"/global/cscratch1/sd/desiproc/dr7/survey-ccds-dr7.kd.fits" $LSLGA_DIR/sample/dr7/

    """
    from legacypipe.survey import LegacySurveyData

    try:
        del survey
    except:
        pass
    
    survey = LegacySurveyData(survey_dir=os.path.join(LSLGAdir, 'sample', dr.lower()),
                                                      output_dir=LSLGAdir)

    return survey


# In[26]:


def simple_wcs(onegroup, diam):
    """Build a simple WCS object for a single group.
    
    """
    from astrometry.util.util import Tan
    
    size = np.rint(diam * 60 / PIXSCALE).astype('int') # [pixels]
    wcs = Tan(onegroup['ra'], onegroup['dec'], size/2+0.5, size/2+0.5,
                 -PIXSCALE/3600.0, 0.0, 0.0, PIXSCALE/3600.0, 
                 float(size), float(size))
    return wcs


# In[27]:


def _build_sample_one(args):
    """Wrapper function for the multiprocessing."""
    return build_sample_one(*args)


# In[28]:


def build_sample_one(onegroup, verbose=False):
    """Wrapper function to find overlapping grz CCDs for a given group.
    
    """
    diam = group_diameter(onegroup) # [arcmin]
    
    wcs = simple_wcs(onegroup, diam)
    try:
        #dr = 'dr5'
        dr = 'dr7'
        #print('Looking for {} in {}...'.format(obj['galaxy'], dr.upper()))
        survey = init_survey(dr=dr)
        ccds = survey.ccds_touching_wcs(wcs)
    except:
        #print('Looking for {} in DR6...'.format(obj['galaxy']))
        try:
            dr = 'dr6'
            survey = init_survey(dr=dr)
            ccds = survey.ccds_touching_wcs(wcs)
        except:
            return [None, None]
    
    if ccds:
        # Is there 3-band coverage?
        if 'g' in ccds.filter and 'r' in ccds.filter and 'z' in ccds.filter:
            if verbose:
                print('Group {:08d}: {} CCDs, RA = {:.5f}, Dec = {:.5f}, Diameter={:.4f} arcmin'.format(
                        onegroup['groupid'], len(ccds), onegroup['ra'], onegroup['dec'], diam))
                sys.stdout.flush()
            return [dr, onegroup]
    
    return [None, None]


# In[29]:


def build_sample(groupcat, use_nproc=nproc):
    """Build the full sample with grz coverage in DR6."""

    sampleargs = list()
    for gg in groupcat:
        sampleargs.append( (gg, True) ) # the False refers to verbose=False

    if use_nproc > 1:
        p = multiprocessing.Pool(nproc)
        result = p.map(_build_sample_one, sampleargs)
        p.close()
    else:
        result = list()
        for args in sampleargs:
            result.append(_build_sample_one(args))
            
    # Remove non-matching objects and write out the sample
    rr = list(zip(*result))    
    outgroupcat = astropy.table.vstack(list(filter(None, rr[1])))
    outgroupcat['dr'] = list(filter(None, rr[0]))
    print('Found {}/{} objects in the DR6/DR7 footprint.'.format(len(outgroupcat), len(groupcat)))
    
    return outgroupcat


# In[ ]:


#groupsample = build_sample_one(groupcat[1408])


# In[ ]:


#%time groupsample = build_sample(groupcat[:500])


# In[30]:


if rebuild_dr_sample:
    samplelogfile = os.path.join(LSLGAdir, 'sample', 'build-sample-{}.log'.format(drsuffix))
    print('Building the sample.')
    print('Logging to {}'.format(samplelogfile))
    t0 = time.time()
    with open(samplelogfile, 'w') as log:
        with redirect_stdout(log):
            groupsample = build_sample(groupcat)
    print('Found {}/ {} groups in the LS footprint.'.format(len(groupsample), len(groupcat)))
    print('Total time = {:.3f} minutes.'.format( (time.time() - t0) / 60 ) )
else:
    


# #### Cross-reference the group catalog to the parent sample and then write out.

# In[ ]:


sample = parentcat[np.where( np.in1d( parentcat['groupid'], groupsample['groupid']) )[0]]
print('Writing {}'.format(samplefile))
sample.write(samplefile, overwrite=True)

print('Writing {}'.format(groupsamplefile))
groupsample.write(groupsamplefile, overwrite=True)


# In[ ]:


qa_radec_dr(groupcat, groupsample)


# In[ ]:


stop


# In[ ]:


leda[leda['d25'] > 0.5]


# ### Get viewer cutouts of a subset of the groups.

# In[ ]:


sample = astropy.table.Table.read(samplefile)
groupsample = astropy.table.Table.read(groupsamplefile)


# In[ ]:


ww = ['g' in oo for oo in sample['objtype']]
sample[ww]


# In[ ]:


jpgdir = os.path.join(LSLGAdir, 'cutouts', 'jpg')
if not os.path.isdir(jpgdir):
    os.mkdir(jpgdir)


# In[ ]:


def get_groupname(group):
    return 'group{:08d}-n{:03d}'.format(group['groupid'], group['nmembers'])


# In[ ]:


def get_layer(group):
    if group['dr'] == 'dr6':
        layer = 'mzls+bass-dr6'
    elif group['dr'] == 'dr5':
        layer = 'decals-dr5'
    elif group['dr'] == 'dr7':
        layer = 'decals-dr5'
    return layer


# In[ ]:


def _get_cutouts_one(args):
    """Wrapper function for the multiprocessing."""
    return get_cutouts_one(*args)


# In[ ]:


def get_cutouts_one(group, clobber=False):
    """Get viewer cutouts for a single galaxy."""

    layer = get_layer(group)
    groupname = get_groupname(group)
        
    diam = group_diameter(group) # [arcmin]
    size = np.ceil(diam * 60 / PIXSCALE).astype('int') # [pixels]

    imageurl = '{}/?ra={:.8f}&dec={:.8f}&pixscale={:.3f}&size={:g}&layer={}'.format(
        cutouturl, group['ra'], group['dec'], PIXSCALE, size, layer)
        
    jpgfile = os.path.join(jpgdir, '{}.jpg'.format(groupname))
    cmd = 'wget --continue -O {:s} "{:s}"' .format(jpgfile, imageurl)
    if os.path.isfile(jpgfile) and not clobber:
        print('File {} exists...skipping.'.format(jpgfile))
    else:
        if os.path.isfile(jpgfile):
            os.remove(jpgfile)
        print(cmd)
        os.system(cmd)
    #sys.stdout.flush()
            
    # Get the fraction of masked pixels
    #im = np.asarray( Image.open(jpgfile) )
    #area = np.product(im.shape[:2])
    #fracmasked = ( np.sum(im[:, :, 0] == 32) / area, 
    #               np.sum(im[:, :, 1] == 32) / area,
    #               np.sum(im[:, :, 2] == 32) / area )
    
    return #np.max(fracmasked)


# In[ ]:


def get_cutouts(groupsample, use_nproc=nproc, clobber=False):
    """Get viewer cutouts of the whole sample."""

    cutoutargs = list()
    for gg in groupsample:
        cutoutargs.append( (gg, clobber) )

    if use_nproc > 1:
        p = multiprocessing.Pool(nproc)
        p.map(_get_cutouts_one, cutoutargs)
        p.close()
    else:
        for args in cutoutargs:
            _get_cutouts_one(args)

    return


# In[ ]:


nmin = 2
indx = groupsample['nmembers'] >= nmin
print('Getting cutouts of {} groups with >={} member(s).'.format(np.sum(indx), nmin))
#groupsample['fracmasked'][indx] = get_cutouts(groupsample[indx], clobber=True)#, use_nproc=1)


# In[ ]:


#get_cutouts_one(groupsample[0], clobber=False)


# In[ ]:


cutlogfile = os.path.join(LSLGAdir, 'cutouts', 'get-cutouts-{}.log'.format(drsuffix))
print('Getting viewer cutouts.')
print('Logging to {}'.format(cutlogfile))
t0 = time.time()
with open(cutlogfile, 'w') as log:
    with redirect_stdout(log):
        get_cutouts(groupsample[indx], clobber=False)
print('Total time = {:.3f} minutes.'.format( (time.time() - t0) / 60 ))


# #### Add labels and a scale bar.

# In[ ]:


barlen = np.round(60.0 / PIXSCALE).astype('int') # [1 arcmin in pixels]
fonttype = os.path.join(LSLGAdir, 'cutouts', 'Georgia.ttf')


# In[ ]:


def get_galaxy(group, sample, html=False):
    """List the galaxy name.
    
    """
    these = group['groupid'] == sample['groupid']
    galaxy = [gg.decode('utf-8').strip().lower() for gg in sample['galaxy'][these].data]
    
    if html:
        galaxy = ' '.join(np.sort(galaxy)).upper()
    else:
        galaxy = ' '.join(np.sort(galaxy))

    return galaxy


# In[ ]:


def _add_labels_one(args):
    """Wrapper function for the multiprocessing."""
    return add_labels_one(*args)


# In[ ]:


def add_labels_one(group, sample, clobber=False, nothumb=False):

    jpgdir = os.path.join(LSLGAdir, 'cutouts', 'jpg')
    pngdir = os.path.join(LSLGAdir, 'cutouts', 'png')
    if not os.path.isdir(pngdir):
        os.mkdir(pngdir)

    groupname = get_groupname(group)
    galaxy = get_galaxy(group, sample, html=True)

    jpgfile = os.path.join(jpgdir, '{}.jpg'.format(groupname))
    pngfile = os.path.join(pngdir, '{}.png'.format(groupname))
    thumbfile = os.path.join(pngdir, 'thumb-{}.png'.format(groupname))
    
    if os.path.isfile(jpgfile):
        if os.path.isfile(pngfile) and not clobber:
            print('File {} exists...skipping.'.format(pngfile))
        else:
            im = Image.open(jpgfile)
            sz = im.size
            fntsize = np.round(sz[0]/28).astype('int')
            width = np.round(sz[0]/175).astype('int')
            font = ImageFont.truetype(fonttype, size=fntsize)
            draw = ImageDraw.Draw(im)

            # Label the group--
            draw.text((0+fntsize*2, 0+fntsize*2), galaxy, font=font)
    
            # Add a scale bar--
            x0, x1, yy = sz[1]-fntsize*2-barlen, sz[1]-fntsize*2, sz[0]-fntsize*2
            draw.line((x0, yy, x1, yy), fill='white', width=width)
            im.save(pngfile)    
        
            # Generate a thumbnail
            if not nothumb:
                cmd = 'convert -thumbnail 300x300 {} {}'.format(pngfile, thumbfile)
                os.system(cmd)


# In[ ]:


def add_labels(groupsample, sample, clobber=False):
    labelargs = list()
    for group in groupsample:
        labelargs.append((group, sample, clobber))
    if nproc > 1:
        p = multiprocessing.Pool(nproc)
        res = p.map(_add_labels_one, labelargs)
        p.close()
    else:
        for args in labelargs:
            res = _add_labels_one(args)


# In[ ]:


get_ipython().run_line_magic('time', 'add_labels(groupsample, sample, clobber=True)')


# In[ ]:


#add_labels_one(groupsample[10], sample, clobber=True)


# ### Finally, assemble the webpage of good and rejected gallery images.
# 
# To test the webpage before release, do
# 
# ```bash
#  rsync -auvP $LLSLGLSLGLSLGALSLGALSLGAA/cutouts/png /global/project/projectdirs/cosmo/www/temp/ioannis/LSLGA/
#  rsync -auvP /global/cscratch1/sd/ioannis/LSLGA/cutouts/*.html /global/project/projectdirs/cosmo/www/temp/ioannis/LSLGA/
# ```
# or
# ```bash
#  rsync -auvP /global/cscratch1/sd/ioannis/LSLGA/cutouts/png /global/project/projectdirs/cosmo/www/temp/ioannis/LSLGA/
#  rsync -auvP /global/cscratch1/sd/ioannis/LSLGA/cutouts/*.html /global/project/projectdirs/cosmo/www/temp/ioannis/LSLGA/
# ```
# and then the website can be viewed here:
#   http://portal.nersc.gov/project/cosmo/temp/ioannis/LSLGA    

# In[ ]:


reject = []
toss = np.zeros(len(groupsample), dtype=bool)
for ii, gg in enumerate(groupsample['groupid']):
    for rej in np.atleast_1d(reject):
        toss[ii] = rej in gg.lower()
        if toss[ii]:
            break
print('Rejecting {} groups.'.format(np.sum(toss)))
groupkeep = groupsample[~toss]
if np.sum(toss) > 0:
    grouprej = groupsample[toss]
else:
    grouprej = []


# In[ ]:


def html_rows(_groupkeep, sample, nperrow=4):
    
    # Not all objects may have been analyzed.
    these = [os.path.isfile(os.path.join(LSLGAdir, 'cutouts', 'png', '{}.png'.format(
        get_groupname(gg)))) for gg in _groupkeep]
    groupkeep = _groupkeep[these]
    
    nrow = np.ceil(len(groupkeep) / nperrow).astype('int')
    groupsplit = list()
    for ii in range(nrow):
        i1 = nperrow*ii
        i2 = nperrow*(ii+1)
        if i2 > len(groupkeep):
            i2 = len(groupkeep)
        groupsplit.append(groupkeep[i1:i2])
    print('Splitting the sample into {} rows with {} mosaics per row.'.format(nrow, nperrow))

    html.write('<table class="ls-gallery">\n')
    html.write('<tbody>\n')
    for grouprow in groupsplit:
        html.write('<tr>\n')
        for group in grouprow:
            groupname = get_groupname(group)
            galaxy = get_galaxy(group, sample, html=True)

            pngfile = os.path.join('cutouts', 'png', '{}.png'.format(groupname))
            thumbfile = os.path.join('cutouts', 'png', 'thumb-{}.png'.format(groupname))
            img = 'src="{}" alt="{}"'.format(thumbfile, galaxy)
            #img = 'class="ls-gallery" src="{}" alt="{}"'.format(thumbfile, nicename)
            html.write('<td><a href="{}"><img {}></a></td>\n'.format(pngfile, img))
        html.write('</tr>\n')
        html.write('<tr>\n')
        for group in grouprow:
            groupname = get_groupname(group)
            galaxy = '{}: {}'.format(groupname.upper(), get_galaxy(group, sample, html=True))
            layer = get_layer(group)
            href = '{}/?layer={}&ra={:.8f}&dec={:.8f}&zoom=12'.format(viewerurl, layer, group['ra'], group['dec'])
            html.write('<td><a href="{}" target="_blank">{}</a></td>\n'.format(href, galaxy))
        html.write('</tr>\n')
    html.write('</tbody>\n')            
    html.write('</table>\n')


# In[ ]:


with open(htmlfile, 'w') as html:
    html.write('<html><head>\n')
    html.write('<style type="text/css">\n')
    html.write('table.ls-gallery {width: 90%;}\n')
    #html.write('img.ls-gallery {display: block;}\n')
    #html.write('td.ls-gallery {width: 100%; height: auto}\n')
    #html.write('td.ls-gallery {width: 100%; word-wrap: break-word;}\n')
    html.write('p.ls-gallery {width: 80%;}\n')
    html.write('</style>\n')
    html.write('</head><body>\n')
    html.write('<h1>Legacy Surveys Large Galaxy Atlas</h1>\n')
    html.write("""<p class="ls-gallery">Each thumbnail links to a larger image while the galaxy 
    name below each thumbnail links to the <a href="http://legacysurvey.org/viewer">Sky Viewer</a>.  
    For reference, the horizontal white bar in the lower-right corner of each image represents 
    one arcminute.</p>\n""")
    #html.write('<h2>Large Galaxy Sample</h2>\n')
    html_rows(groupkeep, sample)
    html.write('</body></html>\n')


# In[ ]:


if len(grouprej) > 0:
    with open(htmlfile_reject, 'w') as html:
        html.write('<html><head>\n')
        html.write('<style type="text/css">\n')
        html.write('img.ls-gallery {display: block;}\n')
        html.write('td.ls-gallery {width: 20%; word-wrap: break-word;}\n')
        html.write('</style>\n')
        html.write('</head><body>\n')
        html.write('<h1>Large Galaxies - Rejected</h1>\n')
        html_rows(grouprej, sample)
        html.write('</body></html>\n')


# In[ ]:


stop


# In[ ]:


#ww = np.where(['NGC' in gg for gg in parent['galaxy']])[0]
#[print(gg.decode('utf-8'), rr, dd) for (gg, rr, dd) in zip(parent['galaxy'][ww].data, parent['ra'][ww].data, parent['dec'][ww].data)]
#print(ww[4])
#parent[ww]

