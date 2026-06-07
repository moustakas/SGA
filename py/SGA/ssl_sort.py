"""
SGA.ssl_sort
============

Generate 128-d UMAP representations for an HDF5 chunk to enable
similarity search via the ssl-legacysurvey pipeline.

Usage:
    python ssl_sort.py chunk.hdf5 --checkpoint resnet50.ckpt --output-dir ./reps
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

from ssl_legacysurvey.utils import load_data
from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.data_analysis import dimensionality_reduction
from ssl_legacysurvey.moco.moco2_module import Moco_v2


def ssl_sort(path, checkpoint_path='resnet50.ckpt', output_dir=None, threshold=0.5):
    """Compute 128-d UMAP representations for a single HDF5 chunk.

    The representations are needed as input to the similarity search
    script.  Results are saved as a .npy file alongside the HDF5.

    Parameters
    ----------
    path : str
        Path to an HDF5 file produced by build_ssl_legacysurvey().
    checkpoint_path : str
        Path to the pre-trained MoCo v2 ResNet-50 checkpoint.
    output_dir : str, optional
        Directory for the output .npy file.
        Defaults to the directory containing `path`.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(path))[0]

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
        'file_head':       'original',
    }

    model    = Moco_v2.load_from_checkpoint(checkpoint_path=checkpoint_path)
    backbone = model.encoder_q
    backbone.fc = torch.nn.Identity()

    transform = datamodules.DecalsTransforms(params['augmentations'], params)
    dataset   = datamodules.DecalsDataset(path, None, transform, params)

    ngals = gals['images'].shape[0]
    im, _ = dataset[0]
    images = torch.empty((ngals, *im.shape), dtype=im.dtype)
    for i in range(ngals):
        images[i], _ = dataset[i]

    representations = backbone(images)
    if params['gpu']:
        representations = representations.detach()
    representations = representations.numpy()

    reps_128, _ = dimensionality_reduction.umap_transform(
        representations, n_components=128, metric='cosine')

    out_path = os.path.join(
        output_dir,
        f"{params['file_head']}_{0:09d}_{params['ngals_tot']:09d}.npy")
    np.save(out_path, reps_128)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute 128-d UMAP representations for ssl similarity search.')
    parser.add_argument('hdf5_files', nargs='+', help='HDF5 chunk file(s)')
    parser.add_argument('--checkpoint', default='resnet50.ckpt', help='Path to resnet50.ckpt')
    parser.add_argument('--output-dir', default=None, help='Output directory for .npy files')
    args = parser.parse_args()

    for path in args.hdf5_files:
        ssl_sort(path, checkpoint_path=args.checkpoint, output_dir=args.output_dir)
