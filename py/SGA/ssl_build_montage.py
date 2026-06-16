"""
SGA.ssl_build_montage
=====================

Build a sorted PDF montage of galaxy cutouts within an HDF5 chunk using
the nearest-neighbour similarity matrix produced by ssl_sort.

Usage:
    python ssl_build_montage.py chunk.hdf5 matrix.npy --per-page 100
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ssl_legacysurvey.utils import load_data
from ssl_legacysurvey.utils import plotting_tools as plt_tools


def greedy_traversal(neighbors: np.ndarray) -> list:
    """Return a visit order that greedily follows nearest neighbours."""
    N = neighbors.shape[0]
    visited = set()
    order = []
    stack = [0]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        order.append(current)
        for neighbor in reversed(neighbors[current]):
            if neighbor not in visited:
                stack.append(neighbor)
        if len(visited) % 5000 == 0:
            print(f'{len(visited)} / {N} galaxies ordered...')
    return order


def build_montage(data_path, matrix_path, output_dir='montages', per_page=100):
    """Build a sorted PDF montage from a similarity matrix.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 chunk (produced by build_ssl_legacysurvey()).
    matrix_path : str
        Path to the .npy nearest-neighbour index matrix.
    output_dir : str
        Directory where the output PDF is written.
    per_page : int
        Number of galaxy thumbnails per PDF page.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(data_path))[0]
    output_path = os.path.join(output_dir, f'{base}.pdf')

    DDL  = load_data.DecalsDataLoader(image_dir=data_path, npix_in=152)
    gals = DDL.get_data(-1, fields=DDL.fields_available, npix_out=152)

    inds         = np.load(matrix_path)
    ordered_inds = greedy_traversal(inds)

    images = gals['images'][ordered_inds]
    ra     = gals['ra'][ordered_inds]
    dec    = gals['dec'][ordered_inds]
    rows   = gals['row'][ordered_inds]

    ncol      = int(np.sqrt(per_page))
    npix_show = 96

    with PdfPages(output_path) as pdf:
        for i in range(0, len(ordered_inds), per_page):
            start = i
            end   = min(i + per_page, len(ordered_inds))
            print(f'Rendering page {i // per_page + 1} ({start}–{end})...')

            fig = plt_tools.show_galaxies(
                images[start:end], ra[start:end], dec[start:end], rows[start:end],
                display_radec=False, display_ref=True,
                nx=ncol, nplt=end - start, npix_show=npix_show)

            pdf.savefig(fig)
            plt.close(fig)

    print(f'Wrote {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a sorted pdf montage from ssl similarity results.')
    parser.add_argument('data_path',   help='HDF5 chunk file')
    parser.add_argument('matrix_path', help='.npy nearest-neighbour index matrix')
    parser.add_argument('--output-dir', default='montages', help='Output directory for PDF')
    parser.add_argument('--per-page', type=int, default=100, help='Thumbnails per page')
    args = parser.parse_args()

    build_montage(args.data_path, args.matrix_path,
                  output_dir=args.output_dir, per_page=args.per_page)
