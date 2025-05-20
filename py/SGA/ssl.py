'''
This script generates a UMAP for the given chunk and applies kmeans clustering in order to sort the candidate cutouts and classify the large galaxies. By default, the script saves the UMAP coordinates and a text file with data for the reject galaxies. Also has been modified so that it can also generate file needed to run similarity search. 
'''

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


def ssl_match(path, similarity=False, threshold=0.5):
    '''
    path: path to the chunk we are running on

    similarity: If we are going to run the similarity search script based on this data, we save the extra file we need (note takes ~twice as long to run)

    threshold: the maximum amount of known galaxies that can be improperly classified before determining that the clustering algorithm did not behave as intended
    on the dataset, in case we don't always trust the clustering. 
    '''

    # Load h5py file into dictionary
    data_path = path
    with h5py.File(path) as F:
        nref = np.sum(F['ref'])

    output_name = os.path.basename(path).replace('.hdf5', '.txt')

    if os.path.exists(output_name): #if we generated the final UMAP already, don't redo it
        print(f"Output file for {path} already exists.")
        return


    #### load in the individual cutout data
    DDL = load_data.DecalsDataLoader(image_dir=data_path, npix_in=152)
    gals = DDL.get_data(-1, fields=DDL.fields_available,npix_out=152) # -1 to load all galaxies


    
    ###### Load in the necessary parameters and the pretrained model
    
    class Args: 
        data_path = path
        base = os.path.splitext(os.path.basename(data_path))[0]
        gpu = True # Use GPU?
        gpus = 1 # Number of gpus to use
        num_nodes = 1
        ngals_tot = gals['images'].shape[0]
        # Read in the checkpoint file from when the SSl model was originally trained
        checkpoint_path = 'resnet50.ckpt'
        rep_dir = os.path.join(base, 'representations/compiled')
        file_head = 'original'
       
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
    # ngals can be subset of the dataset or the whole dataset, typically we use full set when clustering
    ngals =  gals['images'].shape[0]
    im, label = decals_dataloader.__getitem__(0)
    images = torch.empty((ngals, im.shape[0], im.shape[1], im.shape[2]), dtype=im.dtype)
    for i in range(ngals):
        images[i], _ = decals_dataloader.__getitem__(i)

    ######### Run the image cutouts through the pretrained model
    representations = backbone(images)
    
    if params['gpu']: #fix data type to numpy arrays
        images, representations = images.detach(), representations.detach()
    images, representations = images.numpy(), representations.numpy()
    
    if similarity:
        ###### smash down to 128 dimensions instead of 2048, since this is what we will need for similarity script
        reps_128, reps_128_trans = dimensionality_reduction.umap_transform(representations, n_components=128, metric='cosine')
        np.save(f"/pscratch/sd/s/sgmoore1/ssl-legacysurvey/{params['rep_dir']}/{params['file_head']}_{0 :09d}_{params['ngals_tot']:09d}.npy", reps_128)

        #### then smash to 2 dimensions since this is what we need for clustering
        umap_embedding_cos, umap_trans_cos = dimensionality_reduction.umap_transform(reps_128, n_components=2, metric='cosine')

    else: 
        ### if we don't need to run similarity script, just run the dimensionality reduction all at once
        umap_embedding_cos, umap_trans_cos = dimensionality_reduction.umap_transform(representations, n_components=2, metric='cosine')


    #### save the final umap embedding so we can reference it later
    umap_coords = np.array(umap_embedding_cos)
    np.save(f"umap_{params['base']}.npy", umap_coords)

    ####### Separate the data into 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(umap_embedding_cos) #if you are applying the umap first
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

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
    ##### Only compute if we have determined that the known galaxies are well behaved within the clustering
        candidate_coords = []
        rows = [] #temporary storing of the rows within this file
        gals['row'] = gals['row'].astype(int)

        ###### store all of the galaxies that are 'rejected' i.e. in the cluster that doesn't contain most reference galaxies
        i = 0
        while i < len(cluster_labels):
            if int(cluster_labels[i]) != int(gal_cluster):
                if i < int(nref):
                    candidate_coords.append([int(gals['row'][i]), float(gals['ra'][i]), float(gals['dec'][i]), int(1)])
                else: 
                    candidate_coords.append([int(gals['row'][i]), float(gals['ra'][i]), float(gals['dec'][i]), int(0)])
                rows.append(i)
            i += 1
        
        # Convert list to a numpy array and save
        candidate_coords = np.array(candidate_coords)
        np.savetxt(output_name, candidate_coords, delimiter=',', header='ROW, RA,DEC, REF',fmt='%d,%f,%f,%d')
        print(output_name, 'created.')
        
