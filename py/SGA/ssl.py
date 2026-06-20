"""
SGA.ssl
=======

Build ssl-legacysurvey input catalogs from SGA-2025 mosaics, and run
inference with the pre-trained MoCo v2 model.

Typical workflow after generating rescaled cutouts
(SGA2025-cutouts --rescale --region=<region>):

    from astropy.table import vstack
    from SGA.SGA import read_sample
    from SGA.ssl import build_ssl_legacysurvey_refcat, build_ssl_legacysurvey

    ss_n, fs_n = read_sample(region='dr9-north')
    ss_s, fs_s = read_sample(region='dr11-south')
    sample    = vstack([ss_n, ss_s])
    fullsample = vstack([fs_n, fs_s])

    refcat, cat = build_ssl_legacysurvey_refcat(sample, fullsample, ssl_version='v3')
    build_ssl_legacysurvey(cat, refcat, ssl_version='v3',
                           cutoutdir='/path/to/rescaled/cutouts',
                           outdir='/path/to/hdf5/output')

Then run inference on each HDF5 chunk:

    from SGA.ssl import ssl_match
    ssl_match('ssl-parent-chunk000-v3.hdf5',
              checkpoint_path='/path/to/resnet50.ckpt',
              output_dir='/path/to/results')
"""
import os
import numpy as np
import fitsio
from astropy.table import Table, vstack

from astrometry.libkd.spherematch import match_radec

from SGA.SGA import sga_dir, SAMPLE, get_galaxy_galaxydir
from SGA.ellipse import ELLIPSEMODE
from SGA.io import radec_to_name, get_raslice
from SGA.logger import log


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_coadd_bands(galaxydir, group_prefix, bands_str):
    """Return {band: 2D image} for each .fits.fz coadd found in galaxydir."""
    images = {}
    for band in bands_str:
        fitsfile = os.path.join(galaxydir, f'{group_prefix}-image-{band}.fits.fz')
        if os.path.isfile(fitsfile):
            images[band] = fitsio.read(fitsfile)
        else:
            log.debug(f'Missing {band}-band coadd: {fitsfile}')
    return images


def _rescale_one_band(img, width=152):
    """Lanczos-3 resample a square 2D image to (width, width)."""
    from astrometry.util.util import lanczos3_interpolate

    sh, sw = img.shape
    pscale = width / sw

    center_x = sw // 2
    center_y = sh // 2

    cox = np.arange(width, dtype=np.float64) / pscale
    cox += center_x - cox[width // 2]
    coy = np.arange(width, dtype=np.float64) / pscale
    coy += center_y - coy[width // 2]

    fx, fy = np.meshgrid(cox, coy)
    fx = fx.ravel().astype(np.float32)
    fy = fy.ravel().astype(np.float32)
    ix = (fx + 0.5).astype(np.int32)
    iy = (fy + 0.5).astype(np.int32)
    dx = (fx - ix).astype(np.float32)
    dy = (fy - iy).astype(np.float32)

    out = np.zeros(width * width, np.float32)
    lanczos3_interpolate(ix, iy, dx, dy, [out], [np.ascontiguousarray(img, np.float32)])
    return out.reshape((width, width))


def _load_moco_backbone_and_projector(checkpoint_path, device):
    """Load ResNet50 backbone and MLP projection head from a MoCo v2 checkpoint.

    Returns (backbone, projector) as eval-mode modules on device.  The backbone
    outputs 2048-d avgpool features; the projector maps those to 128-d.
    Bypasses pytorch_lightning entirely to avoid version compatibility issues.
    """
    import torch
    import torchvision.models as tvm

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    backbone_state = {k[len('encoder_q.'):]: v
                      for k, v in state_dict.items()
                      if k.startswith('encoder_q.')}
    backbone = tvm.resnet50(weights=None)
    backbone.fc = torch.nn.Identity()
    backbone.load_state_dict(backbone_state, strict=False)
    backbone = backbone.to(device).eval()

    proj_state = {k[len('encoder_q.fc.'):]: v
                  for k, v in state_dict.items()
                  if k.startswith('encoder_q.fc.')}
    projector = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 128),
    )
    projector.load_state_dict(proj_state)
    projector = projector.to(device).eval()

    return backbone, projector


def _get_diam_arcsec(cat):
    """Return diameters in arcsec, handling parent (DIAM) and final (DIAM_INIT) catalogs."""
    if 'DIAM' in cat.colnames:
        return cat['DIAM'].value * 60.
    if 'DIAM_INIT' in cat.colnames:
        return cat['DIAM_INIT'].value * 60.
    raise KeyError('Neither DIAM nor DIAM_INIT found in catalog.')


def _find_fitsfile(ra, dec, cutoutdir, cutout_regions):
    """Return the path to the rescaled FITS cutout for one object, or None."""
    objname = radec_to_name(ra, dec)[0].replace(' ', '_')
    for region in cutout_regions:
        fitsfile = os.path.join(cutoutdir, region, 'rescale',
                                get_raslice(ra), f'{objname}.fits')
        if os.path.isfile(fitsfile):
            return fitsfile
    return None


def _select_band_planes(img, available_bands_str, target_bands):
    """Select planes from a (nband, h, w) image array for the requested bands.

    Returns None if any target band is absent from available_bands_str.
    """
    available = list(available_bands_str)
    try:
        indices = [available.index(b) for b in target_bands]
    except ValueError:
        return None
    return img[indices]


# ---------------------------------------------------------------------------
# Public API: coadd-based cutout rescaling and sample selection
# ---------------------------------------------------------------------------

def rescale_galaxy_cutout(row, galaxydir, width=152):
    """Read and Lanczos-3 resample one SGA group mosaic to (3, width, width).

    Output band order is always (g, r, z).  Missing bands are zero-padded.
    If r is absent but i is present, i is used as a proxy for r.

    Raises FileNotFoundError if no .fits.fz coadds are found.
    """
    group_prefix = row['SGAGROUP']
    raw = _read_coadd_bands(galaxydir, group_prefix, row['BANDS'])

    if not raw:
        raise FileNotFoundError(
            f'No .fits.fz coadds found for {group_prefix} in {galaxydir}')

    if 'r' not in raw and 'i' in raw:
        raw['r'] = raw['i']

    planes = np.zeros((3, width, width), np.float32)
    for iband, b in enumerate(('g', 'r', 'z')):
        if b in raw:
            planes[iband] = _rescale_one_band(raw[b], width)

    return planes


def select_ssl_sample(sample):
    """Return the non-resolved subset of sample for SSL embedding.

    Keeps objects where ELLIPSEMODE & RESOLVED == 0.  No size cut is
    applied; galaxy groups are retained as potentially interesting outliers.
    """
    I = (sample['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED']) == 0
    log.info(f'select_ssl_sample: {I.sum():,d} / {len(sample):,d} objects retained')
    return sample[I]


def _rescale_worker(args):
    sgagroup, bands_str, galaxydir, width = args
    return rescale_galaxy_cutout({'SGAGROUP': sgagroup, 'BANDS': bands_str},
                                 galaxydir, width)


def build_ssl_hdf5(sample, region, datadir, outdir,
                   chunksize=50000, width=152, overwrite=False, mp=1):
    """Pack SGA group mosaics into chunked HDF5 files for SSL inference.

    Parameters
    ----------
    sample : astropy.table.Table
        Non-resolved GROUP_PRIMARY objects from select_ssl_sample().
    region : str
        Survey region, e.g. 'dr11-south'.
    datadir : str
        Root of the SGA data tree ($SGA_DATA_DIR).
    outdir : str
        Output directory; files written as ssl-cutouts-{region}-chunk{N:04d}.hdf5.
    chunksize : int
        Galaxies per HDF5 file.
    width : int
        Output image size in pixels (default 152).
    overwrite : bool
        Overwrite existing output files.
    mp : int
        Worker processes for image rescaling.
    """
    import h5py
    import multiprocessing

    os.makedirs(outdir, exist_ok=True)

    _, galaxydirs = get_galaxy_galaxydir(sample, region=region)

    nobj   = len(sample)
    chunks = np.array_split(np.arange(nobj), max(1, int(np.ceil(nobj / chunksize))))
    log.info(f'build_ssl_hdf5: {nobj:,d} objects → {len(chunks)} chunk(s) in {outdir}')

    for ichunk, idx in enumerate(chunks):
        outfile = os.path.join(outdir, f'ssl-cutouts-{region}-chunk{ichunk:04d}.hdf5')
        if os.path.isfile(outfile) and not overwrite:
            log.info(f'Skipping existing {outfile}')
            continue

        chunk_sample = sample[idx]
        chunk_dirs   = galaxydirs[idx]

        args = [
            (str(row['SGAGROUP']), str(row['BANDS']), str(gdir), width)
            for row, gdir in zip(chunk_sample, chunk_dirs)
        ]

        report_every = max(1000, len(args) // 10)
        images = []
        if mp > 1:
            with multiprocessing.Pool(mp) as P:
                for ii, img in enumerate(P.imap(_rescale_worker, args)):
                    images.append(img)
                    if (ii + 1) % report_every == 0 or ii + 1 == len(args):
                        log.info(f'  chunk {ichunk:04d}: {ii+1:,d} / {len(idx):,d}')
        else:
            for ii, a in enumerate(args):
                images.append(_rescale_worker(a))
                if (ii + 1) % report_every == 0 or ii + 1 == len(args):
                    log.info(f'  chunk {ichunk:04d}: {ii+1:,d} / {len(idx):,d}')

        tmpfile = outfile + '.tmp'
        with h5py.File(tmpfile, 'w') as F:
            F.attrs['region'] = region
            F.create_dataset('images', data=np.stack(images).astype(np.float32),
                             chunks=(1, 3, width, width))
            F.create_dataset('sgaid', data=chunk_sample['SGAID'].value)
            F.create_dataset('ra',    data=chunk_sample['RA'].value)
            F.create_dataset('dec',   data=chunk_sample['DEC'].value)
        os.rename(tmpfile, outfile)
        log.info(f'Wrote {outfile}: {len(idx):,d} objects')


def load_ssl_embeddings(region, ssl_dir, catalog=None):
    """Load SSL embeddings and join to the SGA-2025 catalog on SGAID.

    Parameters
    ----------
    region : str
        Survey region, e.g. 'dr11-south'.
    ssl_dir : str
        Directory containing ssl-embeddings-{region}.hdf5.
    catalog : astropy.table.Table, optional
        Pre-loaded SGA catalog from read_sga_sample(). Read from disk if None.

    Returns
    -------
    astropy.table.Table
        One row per embedded galaxy with all catalog columns plus
        'embeddings' (shape 2048,) and 'projections' (shape 128,).
    """
    import h5py
    from astropy.table import Column

    emb_file = os.path.join(ssl_dir, f'ssl-embeddings-{region}.hdf5')
    if not os.path.isfile(emb_file):
        raise FileNotFoundError(emb_file)

    with h5py.File(emb_file, 'r') as F:
        sgaids      = F['sgaid'][:]
        embeddings  = F['embeddings'][:]
        projections = F['projections'][:]

    if catalog is None:
        catalog, _ = read_sga_sample(region=region)

    cat_lookup = {int(s): i for i, s in enumerate(np.asarray(catalog['SGAID']))}
    found    = np.array([int(s) in cat_lookup for s in sgaids])
    if not np.all(found):
        log.warning(f'{(~found).sum()} embedding SGAIDs not found in catalog; dropping.')
    cat_rows = np.array([cat_lookup[int(s)] for s in sgaids[found]])

    matched = catalog[cat_rows].copy()
    matched.add_column(Column(embeddings[found],  name='embeddings'))
    matched.add_column(Column(projections[found], name='projections'))

    log.info(f'load_ssl_embeddings: {len(matched):,d} objects from {emb_file}')
    return matched


# ---------------------------------------------------------------------------
# Public API: build ssl-legacysurvey input files (legacy viewer-based workflow)
# ---------------------------------------------------------------------------

def build_ssl_legacysurvey_refcat(sample, fullsample, ssl_version=None):
    """Build the ssl-legacysurvey reference and candidate catalogs.

    Parameters
    ----------
    sample : astropy.table.Table
        GROUP_PRIMARY objects from read_sample().
    fullsample : astropy.table.Table
        All group members from read_sample(); used for isolation checks.
    ssl_version : {'v1', 'v2', 'v3'}
        v1 — refcat 1.5–5 arcmin, grz only, fully isolated within 90 arcsec.
        v2 — refcat 1.7–5 arcmin, grz only, no neighbour >30" within 90 arcsec.
        v3 — refcat 1.7–5 arcmin, grz/griz, same neighbour isolation as v2.

    Returns
    -------
    refcat : astropy.table.Table
        Isolated, unresolved reference galaxies (known large galaxies).
    cat : astropy.table.Table
        Smaller unresolved candidate galaxies to classify.
    """
    def find_isolated(cat, fullcat, radius=90.):
        allmatches = match_radec(
            cat['RA'].value, cat['DEC'].value,
            fullcat['RA'].value, fullcat['DEC'].value,
            radius / 3600., indexlist=True, notself=False)
        return np.array([ii for ii, mm in enumerate(allmatches) if len(mm) == 1])

    diam      = _get_diam_arcsec(sample)
    diam_full = _get_diam_arcsec(fullsample)
    is_resolved = (sample['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED']) != 0
    not_nearstar = (sample['SAMPLE'] & SAMPLE['NEARSTAR']) == 0

    if ssl_version == 'v1':
        has_bands = sample['BANDS'] == 'grz'
        I = np.where((diam / 60. > 1.5) * (diam / 60. < 5.) *
                     ~is_resolved * not_nearstar * has_bands)[0]
        refcat = sample[I[find_isolated(sample[I], fullsample)]]
        cat    = sample[(diam < 10.) * has_bands * ~is_resolved]

    elif ssl_version == 'v2':
        has_bands = sample['BANDS'] == 'grz'
        I = np.where((diam / 60. > 1.7) * (diam / 60. < 5.) *
                     ~is_resolved * not_nearstar * has_bands)[0]
        J = (diam_full > 30.) * (fullsample['BANDS'] == 'grz')
        refcat = sample[I[find_isolated(sample[I], fullsample[J])]]
        cat    = sample[(diam < 30.) * has_bands * ~is_resolved]

    elif ssl_version == 'v3':
        has_grz      = np.isin(sample['BANDS'],    ['grz', 'griz', 'girz'])
        has_grz_full = np.isin(fullsample['BANDS'], ['grz', 'griz', 'girz'])
        I = np.where((diam / 60. > 1.7) * (diam / 60. < 5.) *
                     ~is_resolved * not_nearstar * has_grz)[0]
        J = (diam_full > 30.) * has_grz_full
        refcat = sample[I[find_isolated(sample[I], fullsample[J])]]
        cat    = sample[(diam < 10.) * has_grz * ~is_resolved]

    else:
        raise ValueError(f'Unsupported ssl_version={ssl_version!r}')

    log.info(f'ssl_version={ssl_version}: {len(refcat):,d} reference objects, '
             f'{len(cat):,d} candidates to classify.')
    return refcat, cat


def build_ssl_legacysurvey(cat, refcat, width=152, ncatmax=15000,
                           ssl_version=None, bands=('g', 'r', 'z'),
                           cutoutdir='.', outdir='.', cutout_regions=None,
                           verbose=False, overwrite=False):
    """Pack rescaled FITS cutouts into HDF5 files for ssl-legacysurvey.

    Each output HDF5 file contains a reference set followed by a chunk of
    candidates.  The 'row' dataset stores SGAID values as the unique object
    key (replaces the old ROW_PARENT column).

    Parameters
    ----------
    cat : astropy.table.Table
        Candidate objects from build_ssl_legacysurvey_refcat().
    refcat : astropy.table.Table
        Reference objects from build_ssl_legacysurvey_refcat().
    width : int
        Cutout pixel width; must match the rescaled cutouts.
    ncatmax : int
        Maximum number of candidates per HDF5 chunk.
    ssl_version : str
        Version label for output filenames, e.g. 'v3'.
    bands : sequence of str
        Band planes to store in each chunk, e.g. ('g', 'r', 'z').
    cutoutdir : str
        Root of the rescaled cutout tree ({cutoutdir}/{region}/rescale/...).
    outdir : str
        Root output directory; HDF5 chunks go to {outdir}/{ssl_version}/input/.
    cutout_regions : list of str, optional
        Region subdirectories to search for cutouts, tried in order.
        Defaults to ['dr9-north', 'dr11-south'].
    overwrite : bool
        Overwrite existing output files.
    """
    import h5py

    if ssl_version is None:
        raise ValueError('ssl_version is required.')
    if cutout_regions is None:
        cutout_regions = ['dr9-north', 'dr11-south']

    bands = list(bands)
    nband = len(bands)

    def collect_files(tbl):
        files, keep = [], []
        for ii, row in enumerate(tbl):
            f = _find_fitsfile(row['RA'], row['DEC'], cutoutdir, cutout_regions)
            if f is not None:
                files.append(f)
                keep.append(ii)
            else:
                log.debug(f"Missing cutout for SGAID={row['SGAID']}")
        return np.array(files), np.array(keep, dtype=int)

    refcatfiles, refkeep = collect_files(refcat)
    catfiles,    catkeep = collect_files(cat)
    refcat = refcat[refkeep]
    cat    = cat[catkeep]
    nrefcat, ncat = len(refcat), len(cat)
    log.info(f'{nrefcat:,d} reference and {ncat:,d} candidate objects with cutouts found.')

    catdir = os.path.join(sga_dir(), 'ssl')
    os.makedirs(catdir, exist_ok=True)

    refoutfile = os.path.join(catdir, f'ssl-parent-refcat-{ssl_version}.fits')
    outfile    = os.path.join(catdir, f'ssl-parent-cat-{ssl_version}.fits')
    for fitsout, tbl, label in ((refoutfile, refcat, 'reference'),
                                 (outfile,    cat,    'candidate')):
        if os.path.isfile(fitsout) and not overwrite:
            log.warning(f'Existing {label} catalog {fitsout}; '
                        f'remove it manually or use overwrite=True.')
            return
        tbl.write(fitsout, overwrite=overwrite)
        log.info(f'Wrote {len(tbl):,d} {label} objects to {fitsout}')

    nchunk = max(1, int(np.ceil(ncat / ncatmax)))
    chunks = np.array_split(np.arange(ncat), nchunk)
    log.info(f'Writing {nchunk:,d} HDF5 chunk(s) to {outdir}/{ssl_version}/input/')

    for ichunk, chunk in enumerate(chunks):
        h5dir  = os.path.join(outdir, ssl_version, 'input')
        os.makedirs(h5dir, exist_ok=True)
        h5file = os.path.join(h5dir, f'ssl-parent-chunk{ichunk:03d}-{ssl_version}.hdf5')

        if os.path.isfile(h5file) and not overwrite:
            log.info(f'Skipping existing {h5file}')
            continue

        refs   = np.concatenate([np.ones(nrefcat, bool), np.zeros(len(chunk), bool)])
        sgaids = np.concatenate([refcat['SGAID'].value,   cat['SGAID'][chunk].value])
        ras    = np.concatenate([refcat['RA'].value,       cat['RA'][chunk].value])
        decs   = np.concatenate([refcat['DEC'].value,      cat['DEC'][chunk].value])
        all_files = np.concatenate([refcatfiles,           catfiles[chunk]])
        all_bands = np.concatenate([refcat['BANDS'].value, cat['BANDS'][chunk].value])
        ntot = len(refs)

        n_missing = 0
        with h5py.File(h5file, 'w') as F:
            F.create_dataset('ref', data=refs)
            F.create_dataset('row', data=sgaids)
            F.create_dataset('ra',  data=ras)
            F.create_dataset('dec', data=decs)
            images_ds = F.create_dataset('images', (ntot, nband, width, width),
                                         dtype=np.float32)
            for iobj, (fitsfile, bands_str) in enumerate(zip(all_files, all_bands)):
                img    = fitsio.read(fitsfile)
                planes = _select_band_planes(img, bands_str, bands)
                if planes is None:
                    log.warning(f'Band mismatch for {fitsfile} (has {bands_str!r}); zeroing.')
                    n_missing += 1
                    continue
                images_ds[iobj] = planes.astype(np.float32)

        log.info(f'Wrote {h5file}: {nrefcat:,d} reference + {len(chunk):,d} candidates'
                 + (f' ({n_missing} band mismatches zeroed)' if n_missing else ''))


# ---------------------------------------------------------------------------
# Public API: run ssl-legacysurvey inference
# ---------------------------------------------------------------------------

def extract_embeddings(hdf5_files, checkpoint_path, outfile,
                       batch_size=256, device=None, overwrite=False):
    """Extract ResNet50 embeddings and MoCo projection outputs for all SSL cutout chunks.

    Streams through the HDF5 chunk files produced by build_ssl_hdf5(), runs
    batched GPU/CPU inference, and writes a single output HDF5 containing:

        embeddings  (N, 2048) float32  — backbone avgpool features
        projections (N, 128)  float32  — MLP projection head output
        sgaid       (N,)      int64
        ra          (N,)      float64
        dec         (N,)      float64

    L2-normalize projections before cosine similarity search or Faiss indexing.

    Parameters
    ----------
    hdf5_files : list of str
        Paths to ssl-cutouts-{region}-chunk*.hdf5 files from build_ssl_hdf5().
    checkpoint_path : str
        Path to the pretrained MoCo v2 ResNet50 checkpoint.
    outfile : str
        Output HDF5 path.
    batch_size : int
        GPU batch size.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    overwrite : bool
        Overwrite existing output file.
    """
    import torch
    import h5py

    if os.path.isfile(outfile) and not overwrite:
        log.info(f'Skipping existing {outfile}')
        return

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    log.info(f'Using device: {device}')

    backbone, projector = _load_moco_backbone_and_projector(checkpoint_path, device)
    log.info(f'Loaded backbone + projector from {checkpoint_path}')

    tmpfile = outfile + '.tmp'
    with h5py.File(tmpfile, 'w') as F:
        F.attrs['checkpoint'] = checkpoint_path
        emb_ds   = F.create_dataset('embeddings',  shape=(0, 2048), maxshape=(None, 2048),
                                    dtype=np.float32, chunks=(1000, 2048))
        proj_ds  = F.create_dataset('projections', shape=(0, 128),  maxshape=(None, 128),
                                    dtype=np.float32, chunks=(1000, 128))
        sgaid_ds = F.create_dataset('sgaid', shape=(0,), maxshape=(None,), dtype=np.int64)
        ra_ds    = F.create_dataset('ra',    shape=(0,), maxshape=(None,), dtype=np.float64)
        dec_ds   = F.create_dataset('dec',   shape=(0,), maxshape=(None,), dtype=np.float64)

        n_total = 0
        for ifile, hdf5_file in enumerate(hdf5_files):
            log.info(f'Processing {os.path.basename(hdf5_file)} ({ifile+1}/{len(hdf5_files)})')

            with h5py.File(hdf5_file, 'r') as H:
                images = H['images'][:]
                sgaids = H['sgaid'][:]
                ras    = H['ra'][:]
                decs   = H['dec'][:]

            n = len(images)
            chunk_embs  = np.zeros((n, 2048), np.float32)
            chunk_projs = np.zeros((n, 128),  np.float32)

            report_every = max(batch_size, n // 10)
            last_reported = -1
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    batch = torch.from_numpy(images[start:end]).to(device)
                    emb = backbone(batch)
                    chunk_embs[start:end]  = emb.cpu().numpy()
                    chunk_projs[start:end] = projector(emb).cpu().numpy()
                    milestone = end // report_every
                    if milestone != last_reported:
                        log.info(f'  {end:,d} / {n:,d}')
                        last_reported = milestone

            emb_ds.resize(n_total + n, axis=0)
            proj_ds.resize(n_total + n, axis=0)
            sgaid_ds.resize(n_total + n, axis=0)
            ra_ds.resize(n_total + n, axis=0)
            dec_ds.resize(n_total + n, axis=0)

            emb_ds[n_total:n_total+n]   = chunk_embs
            proj_ds[n_total:n_total+n]  = chunk_projs
            sgaid_ds[n_total:n_total+n] = sgaids
            ra_ds[n_total:n_total+n]    = ras
            dec_ds[n_total:n_total+n]   = decs

            n_total += n
            log.info(f'  cumulative: {n_total:,d} objects')

    os.rename(tmpfile, outfile)
    log.info(f'Wrote {outfile}: {n_total:,d} objects')


def ssl_match(path, checkpoint_path='resnet50.ckpt', output_dir=None,
              similarity=False, threshold=0.5):
    """Run ssl-legacysurvey inference on one HDF5 chunk.

    Loads images from the HDF5 file, passes them through the pre-trained
    MoCo v2 backbone, computes a 2-d UMAP embedding, and uses k-means
    clustering to separate galaxy candidates from non-galaxies.  The
    reference objects (ref=True in the HDF5) anchor which cluster is
    the "galaxy" cluster.

    Parameters
    ----------
    path : str
        Path to an HDF5 file produced by build_ssl_legacysurvey().
    checkpoint_path : str
        Path to the pre-trained MoCo v2 ResNet-50 checkpoint.
    output_dir : str, optional
        Directory for the output .txt and umap .npy files.
        Defaults to the directory containing `path`.
    similarity : bool
        If True, also save 128-d UMAP representations for a similarity search.
    threshold : float
        Maximum fraction of misclassified reference objects before flagging
        a clustering failure (default 0.5).
    """
    import h5py
    import torch
    from ssl_legacysurvey.utils import load_data
    from ssl_legacysurvey.data_loaders import datamodules
    from ssl_legacysurvey.data_analysis import dimensionality_reduction
    from ssl_legacysurvey.moco.moco2_module import Moco_v2
    from sklearn.cluster import KMeans

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(path))

    with h5py.File(path) as F:
        nref = int(np.sum(F['ref']))

    base       = os.path.splitext(os.path.basename(path))[0]
    output_txt  = os.path.join(output_dir, f'{base}.txt')
    output_umap = os.path.join(output_dir, f'umap_{base}.npy')

    if os.path.exists(output_txt):
        log.info(f'Output already exists: {output_txt}')
        return

    DDL  = load_data.DecalsDataLoader(image_dir=path, npix_in=152)
    gals = DDL.get_data(-1, fields=DDL.fields_available, npix_out=152)

    params = {
        'data_path':       path,
        'base':            base,
        'gpu':             torch.cuda.is_available(),
        'gpus':            1,
        'num_nodes':       1,
        'ngals_tot':       gals['images'].shape[0],
        'checkpoint_path': checkpoint_path,
        'ssl_training':    False,
        'jitter_lim':      0,
        'augmentations':   'jcrg',
    }

    model    = Moco_v2.load_from_checkpoint(checkpoint_path=checkpoint_path)
    backbone = model.encoder_q
    backbone.fc = torch.nn.Identity()

    transform = datamodules.DecalsTransforms(params['augmentations'], params)
    dataset   = datamodules.DecalsDataset(path, None, transform, params)

    ngals = gals['images'].shape[0]
    im0, _ = dataset[0]
    images = torch.empty((ngals, *im0.shape), dtype=im0.dtype)
    for i in range(ngals):
        images[i], _ = dataset[i]

    representations = backbone(images)
    if params['gpu']:
        representations = representations.detach()
    representations = representations.numpy()

    if similarity:
        reps_128, _ = dimensionality_reduction.umap_transform(
            representations, n_components=128, metric='cosine')
        np.save(os.path.join(output_dir, f'reps128_{base}.npy'), reps_128)
        umap_input = reps_128
    else:
        umap_input = representations

    umap_coords, _ = dimensionality_reduction.umap_transform(
        umap_input, n_components=2, metric='cosine')
    np.save(output_umap, umap_coords)

    labels        = KMeans(n_clusters=2, random_state=0).fit(umap_coords).labels_
    avg_ref_label = np.mean(labels[:nref])
    if avg_ref_label < threshold:
        gal_cluster = 0
    elif avg_ref_label > (1 - threshold):
        gal_cluster = 1
    else:
        log.warning(f'{path}: clustering unstable — >{threshold:.0%} of reference '
                    f'objects misclassified.  Skipping output.')
        return

    gals['row'] = gals['row'].astype(int)
    candidates = np.array([
        [int(gals['row'][i]), float(gals['ra'][i]), float(gals['dec'][i]), int(i < nref)]
        for i in range(len(labels))
        if int(labels[i]) != gal_cluster
    ])
    np.savetxt(output_txt, candidates, delimiter=',',
               header='ROW,RA,DEC,REF', fmt='%d,%f,%f,%d')
    log.info(f'Wrote {len(candidates):,d} candidates to {output_txt}')
