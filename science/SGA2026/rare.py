"""
rare
====

Shared helpers for the rare-galaxy search notebooks (see
rare-galaxies-plan.md). The image-display functions are adapted from
doc/tutorials/SGA-ssl.ipynb.

"""
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

from SGA.qa import sdss_rgb

VIEWER_LAYER = {
    'dr11-south': 'ls-dr11-south',
    'dr11-north': 'ls-dr11-north',
}


def build_cutout_index(ssl_dir, region):
    """Map each SGAID to its location in the cutout HDF5 files.

    Parameters
    ----------
    ssl_dir : str
        Directory containing ssl-cutouts-{region}-chunk*.hdf5.
    region : str
        Survey region, e.g. 'dr11-south'.

    Returns
    -------
    dict
        {sgaid: (hdf5_path, row_index)}.

    """
    files = sorted(glob.glob(os.path.join(ssl_dir, f'ssl-cutouts-{region}-chunk*.hdf5')))
    if not files:
        raise FileNotFoundError(f'No cutout files found in {ssl_dir}')
    index = {}
    for f in files:
        with h5py.File(f, 'r') as H:
            for i, sgaid in enumerate(H['sgaid'][:]):
                index[int(sgaid)] = (f, i)
    return index


def show_cutout_grid(sgaids, cutout_index, ncols=5, figsize_per=2, titles=None):
    """Display a grid of grz cutouts.

    Parameters
    ----------
    sgaids : sequence of int
        Galaxies to show, in order.
    cutout_index : dict
        Output of build_cutout_index.
    ncols : int
        Number of columns in the grid.
    figsize_per : float
        Size of each panel in inches.
    titles : sequence of str, optional
        Panel titles. Defaults to the SGAID.

    """
    sgaids = [int(s) for s in sgaids]
    nrows = int(np.ceil(len(sgaids) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * figsize_per, nrows * figsize_per))
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, sgaid) in enumerate(zip(axes, sgaids)):
        if sgaid in cutout_index:
            fname, idx = cutout_index[sgaid]
            with h5py.File(fname, 'r') as H:
                img = H['images'][idx]
            rgb = sdss_rgb([img[0], img[1], img[2]], ['g', 'r', 'z'])
            ax.imshow(rgb, origin='lower')
        else:
            ax.text(0.5, 0.5, 'not found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
        title = str(titles[i]) if titles is not None else str(sgaid)
        ax.set_title(title, fontsize=max(6, int(figsize_per * 4)))

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()


def print_viewer_links(sgaids, data, region, zoom=14):
    """Print clickable Legacy Survey viewer links.

    Parameters
    ----------
    sgaids : sequence of int
        Galaxies to link.
    data : astropy.table.Table
        Catalog with SGAID, RA, DEC, and OBJNAME columns.
    region : str
        'dr11-south' or 'dr11-north'.
    zoom : int
        Viewer zoom level.

    """
    from IPython.display import display, HTML
    layer = VIEWER_LAYER[region]
    row_of = {int(s): i for i, s in enumerate(np.asarray(data['SGAID']))}
    rows = []
    for sgaid in sgaids:
        i = row_of.get(int(sgaid))
        if i is None:
            continue
        ra, dec = float(data['RA'][i]), float(data['DEC'][i])
        name = str(data['OBJNAME'][i])
        url = (f'https://www.legacysurvey.org/viewer-dev/'
               f'?ra={ra:.6f}&dec={dec:.6f}&layer={layer}'
               f'&sga2025-parent&zoom={zoom}')
        rows.append(f'<li><a href="{url}" target="_blank">{name} (SGAID {sgaid})</a></li>')
    display(HTML('<ul>' + ''.join(rows) + '</ul>'))


def normalize(vecs):
    """L2-normalize each row, so that a dot product is a cosine similarity.

    Parameters
    ----------
    vecs : array, shape (N, D)
        Embedding vectors.

    Returns
    -------
    array, shape (N, D), float32

    """
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)


def rank_by_similarity(vecs, query_rows):
    """Rank every galaxy by its mean cosine similarity to a set of queries.

    Parameters
    ----------
    vecs : array, shape (N, D)
        L2-normalized vectors (see normalize).
    query_rows : sequence of int
        Row indices of the query galaxies.

    Returns
    -------
    order : array of int
        Row indices sorted from most to least similar, queries excluded.
    score : array, shape (N,)
        Mean cosine similarity to the queries, in the original row order.

    """
    query_rows = np.atleast_1d(query_rows)
    score = (vecs @ vecs[query_rows].T).mean(axis=1)
    order = np.argsort(score)[::-1]
    order = order[~np.isin(order, query_rows)]
    return order, score


def knn(vecs, k=10):
    """Find the k nearest neighbors of every galaxy using Faiss.

    Parameters
    ----------
    vecs : array, shape (N, D)
        L2-normalized vectors (see normalize).
    k : int
        Number of neighbors, not counting the galaxy itself.

    Returns
    -------
    sims : array, shape (N, k)
        Cosine similarities, most similar first.
    rows : array, shape (N, k)
        Row indices of the neighbors.

    Notes
    -----
    This is an exact, all-versus-all search. It is fast for the 128-d
    projections; for the 2048-d embeddings, reduce the dimension with PCA
    first or work with a subsample.

    """
    import faiss
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    sims, rows = index.search(vecs, k + 1)
    return sims[:, 1:], rows[:, 1:]


def recall_curve(order, is_target):
    """Fraction of known targets recovered versus candidates inspected.

    Parameters
    ----------
    order : array of int
        Row indices sorted from best to worst candidate.
    is_target : array of bool, shape (N,)
        True for the held-out known examples (e.g., known rings).

    Returns
    -------
    ninspected : array of int
        1, 2, 3, ... candidates inspected.
    recall : array of float
        Fraction of the targets found among the first ninspected candidates.

    """
    hits = np.asarray(is_target)[order]
    ninspected = np.arange(1, len(order) + 1)
    return ninspected, np.cumsum(hits) / max(1, hits.sum())


def record_vi(sgaids, label, vifile):
    """Append visual-inspection classifications to a CSV file.

    Parameters
    ----------
    sgaids : sequence of int
        Galaxies being classified.
    label : str
        Class assigned to all of them (e.g., 'ring', 'not-ring', 'artifact').
    vifile : str
        Output CSV; created with a header if it does not exist.

    """
    from datetime import date
    new = not os.path.isfile(vifile)
    with open(vifile, 'a') as F:
        if new:
            F.write('sgaid,label,date\n')
        for sgaid in sgaids:
            F.write(f'{int(sgaid)},{label},{date.today().isoformat()}\n')
