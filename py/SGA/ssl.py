######## A function designed to output a list of RA and DEC coordinates for candidates of new SGA galaxies.
import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import h5py

import numpy as np
import torch
from ssl_legacysurvey.utils import load_data # Loading galaxy catalogue and image data from hdf5 file(s)
from ssl_legacysurvey.utils import plotting_tools as plt_tools # Plotting images or catalogue info

from ssl_legacysurvey.data_loaders import datamodules # Pytorch dataloaders and datamodules
from ssl_legacysurvey.data_loaders import decals_augmentations # Augmentations for training

from ssl_legacysurvey.data_analysis import dimensionality_reduction # PCA/UMAP functionality
import matplotlib.pyplot as plt
# import cuml
# from cuml.manifold.umap import UMAP
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import h5py

import argparse
# import logging

from pathlib import Path
import os
import sys
import glob
import math
# import astropy
# from astropy.io import fits

from pytorch_lightning import loggers as pl_loggers
# from pl_bolts.models.self_supervised import Moco_v2
from pytorch_lightning.plugins import DDPPlugin
#strategy = 'ddp'
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


def ssl_match(path, threshold=0.1, show_candidates=False):
    ##### The threshold represents the maximum amount of known galaxies that can be improperly classified before determining that the clustering algorithm did not behave as intended on the dataset.

    # Load h5py file into dictionary
    data_path = path
    with h5py.File(path) as F:
        nref = np.sum(F['ref'])

    output_name = os.path.basename(path).replace('.hdf5', '.txt')

    
    DDL = load_data.DecalsDataLoader(image_dir=data_path, npix_in=152)
    
    gals = DDL.get_data(-1, fields=DDL.fields_available,npix_out=152) # -1 to load all galaxies

    ###### Load in the necessary parameters and load the model.
    
    class Args: # In general codes in this project use argparse. Args() simplifies this for this example 
        # Data location and GPU availability 
        # Data path should be the SAME as above
        data_path = path
        gpu = True # Use GPU?
        gpus = 1 # Number of gpus to use
        num_nodes = 1
        ngals_tot = gals['images'].shape[0]
        # Read in the checkpoint file from when the SSl model was originally trained
        checkpoint_path = 'resnet50.ckpt'
       
    params = vars(Args)
    p = {}
    for k, v in params.items():
        p[k] = v
    params = p
    
    model = Moco_v2.load_from_checkpoint(
            checkpoint_path=params['checkpoint_path']
            )
      
    backbone = model.encoder_q
    
    # Remove the MLP projection head from the model, so output is now the representaion for each galaxy
    backbone.fc = torch.nn.Identity()
    
    params['ssl_training'] = False
    params['jitter_lim'] = 0 #determines the maximum value for the offset in the jitter augmentation
    params['augmentations'] = 'jcrg' #adjust to whatever parameters you want, typically 'jcrg' or 'rrjc'
    
    # Load all images as one batch
    
    transform = datamodules.DecalsTransforms(
        params['augmentations'],
        params
    )
    
    
    decals_dataloader = datamodules.DecalsDataset(
        data_path,
        None,
        transform,
        params,
    )
    # ngals can be subset of the dataset or the whole dataset
    ngals =  gals['images'].shape[0]
    im, label = decals_dataloader.__getitem__(0)
    images = torch.empty((ngals, im.shape[0], im.shape[1], im.shape[2]), dtype=im.dtype)
    for i in range(ngals):
        images[i], _ = decals_dataloader.__getitem__(i)

    ######### Run the image cutouts through the pretrained model
    
    representations = backbone(images)
    
    
    if params['gpu']:
        images, representations = images.detach(), representations.detach()
    images, representations = images.numpy(), representations.numpy()
    
    from ssl_legacysurvey.data_analysis import dimensionality_reduction
    from ssl_legacysurvey.utils import plotting_tools as plt_tools # Plotting images or catalogue info
    
    ####### Generate the umap embedding, for now we are using cosine but there are others like Euclidean that didn't make much difference.
    umap_embedding_cos, umap_trans_cos = dimensionality_reduction.umap_transform(representations, n_components=2, metric='cosine')


    ####### Separate the data into 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(umap_embedding_cos) #if you are applying the umap first, otherwise use representations or some other value here
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    print('Did clustering')

    ###### Determine which cluster label corresponds to the cluster with known galaxies in it by finding the average of the cluster label of known galaxies and seeing if that value is closer to 0 or 1. Assumes that all of the known galaxies are at the beginning of the dataset.
    average = np.mean(cluster_labels[:int(nref)])
    print(average)
    if average < threshold:
        gal_cluster = 0
    elif average > (1-threshold):
        gal_cluster = 1
    else:
        print('Error: More known galaxies than the threshold are not properly classified by this clustering algorithm.')
        gal_cluster = 2
    
    if gal_cluster < 2:
        print('okay')
    ##### Limit the resulting output to exclude all of the input galaxies that are known as being within the SGA so that only new candidates are shown. Also, only compute if we have determined that the known galaxies are well behaved within the clustering
        candidate_coords = []
        rows = [] #temporary storing of the rows within this file
        print(gals['row'][:3])
        gals['row'] = gals['row'].astype(int)
        print(gals['row'][:3])
        
        i = 0
        while i < len(cluster_labels):
            if int(cluster_labels[i]) != int(gal_cluster):
                if i < int(nref):
                    candidate_coords.append([int(gals['row'][i]), float(gals['ra'][i]), float(gals['dec'][i]), int(1)])
                else: 
                    candidate_coords.append([int(gals['row'][i]), float(gals['ra'][i]), float(gals['dec'][i]), int(0)])
                rows.append(i)
            i += 1
        
        # Convert list to a numpy array for saving
        candidate_coords = np.array(candidate_coords)
        
        print(candidate_coords)
        np.savetxt(output_name, candidate_coords, delimiter=',', header='ROW, RA,DEC, REF',fmt='%d,%f,%f,%d')

    if show_candidates == True:
        
        plt_tools.show_galaxies(gals['images'][rows[i].astype(int)],
                            candidate_coords[1], candidate_coords[2], 
                            nx=20, nplt=int(len(images)), npix_show=96)
        
    
# def show_candidates(ra, dec, npix_show=152, ncol=10, save=False, output_filename='show_candidates_test'):
#     ######## Creates a grid displaying cutouts of 
#     i=0
#     images = []
#     while i < len(ra):
#         f1 = math.trunc(ra[i])
#         f2 = math.trunc((1000*ra[i]))
#         f3 = math.trunc((100000*ra[i]))
#         f4 = 'm' if dec[i] < 0 else 'p'
#         f5 = abs(math.trunc((100000*dec[i])))
        
#         img = fits.open('/pscratch/sd/s/sgmoore1/SGA2024/cutouts/{:03d}/{:06d}/{:08d}{}{:07d}.fits'.format(f1,f2,f3,f4,f5))
        
#         images[i] = np.array(img[0].data)
#         i+=1
#     # Display images of galaxy candidates in grid, with ra/dec for each galaxy. Colors are after grb transformation, not raw data

#     plt_tools.show_galaxies(images,
#                             ra, dec, 
#                             nx=ncol, nplt=int(len(images)), npix_show=npix_show)
#     if save:
#         plt.savefig('{}.png'.format(output_filename))

# ssl_match(path = '/pscratch/sd/i/ioannis/SGA2025/cutouts/parent/ssl-dr9-north-v1.hdf5')
