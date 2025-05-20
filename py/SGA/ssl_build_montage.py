''' 
This function builds a sorted montage of all of the galaxy cutouts within a specified chunk based on the similarity score scripts.

'''

####### import all of the potentially needed packages
import os
import h5py
import numpy as np
import torch
from ssl_legacysurvey.utils import load_data # Loading galaxy catalogue and image data from hdf5 file(s)
from ssl_legacysurvey.utils import plotting_tools as plt_tools # Plotting images or catalogue info
from ssl_legacysurvey.data_loaders import datamodules # Pytorch dataloaders and datamodules
from ssl_legacysurvey.data_loaders import decals_augmentations # Augmentations for training
from ssl_legacysurvey.data_analysis import dimensionality_reduction # PCA/UMAP functionality
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import h5py
import argparse
from pathlib import Path
import os
import sys
import glob
import math
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from ssl_legacysurvey.moco.moco2_module import Moco_v2 
from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.utils import format_logger
from scripts import predict
from ssl_legacysurvey.finetune import extract_model_outputs
from scripts import similarity_search_nxn
from sklearn.cluster import KMeans
import psutil
import os
import time
from matplotlib.backends.backend_pdf import PdfPages


######### function used to generate the universal ordering based on the nearest neigbors matrix
def greedy_traversal(neighbors: np.ndarray) -> list:
    N = neighbors.shape[0]
    visited = set()
    order = []

    stack = [0]  # start traversal from galaxy 0

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        visited.add(current)
        order.append(current)

        # Add unvisited neighbors to stack in reverse order
        # so first neighbor is popped first
        for neighbor in reversed(neighbors[current]):
            if neighbor not in visited:
                stack.append(neighbor)

        #Display progress bar
        if len(visited) % 5000 == 0:
            print(f"{len(visited)} / {N} galaxies ordered...")

    return order


def main(data_path, matrix_path, per_pg =100):
    ##### generate output filename
    base = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = 'montages'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base}.pdf")

    ### load in the galaxies data
    DDL = load_data.DecalsDataLoader(image_dir=data_path, npix_in=152)
    gals = DDL.get_data(-1, fields=DDL.fields_available,npix_out=152) # -1 to load all galaxies

    ####load in the indices from the matrix data
    inds = np.load(matrix_path)

    ##### sort the indices based on nearest neighbors
    ordered_inds = greedy_traversal(inds)

    ##### gather data for each of the ordered indices
    images = gals['images'][ordered_inds] 
    ra = gals['ra'][ordered_inds]
    dec = gals['dec'][ordered_inds]
    rows = gals['row'][ordered_inds]

    ##### generate the pdf
    images_per_page = per_pg
    ncol = int(np.sqrt(images_per_page))
    npix_show = 96
    with PdfPages(output_path) as pdf:
        for i in range(0, len(ordered_inds), images_per_page):
            start = i
            end = min(i + images_per_page, len(ordered_inds))

            print(f"Rendering page {i // images_per_page + 1} with galaxies {start} to {end}...")

            # Create figure
            fig = plt_tools.show_galaxies(images[start:end],
                                      ra[start:end], dec[start:end], rows[start:end],
                                      display_radec=False, display_ref=True,
                                      nx=ncol, nplt=end - start, npix_show=npix_show)

            pdf.savefig(fig)
            plt.close(fig)
    
