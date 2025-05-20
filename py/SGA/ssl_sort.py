'''
This script generates and saves the file that is needed to run the similarity search script.
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

def ssl_match(path, threshold=0.5):
    
    # Load h5py file into dictionary
    data_path = path
    # with h5py.File(path) as F: 
    #     nref = np.sum(F['ref'])

    ##### extract out 
    DDL = load_data.DecalsDataLoader(image_dir=data_path, npix_in=152)
    gals = DDL.get_data(-1, fields=DDL.fields_available,npix_out=152) # -1 to load all galaxies

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
    
    params = dict(vars(Args))
    p = {}
    for k, v in params.items():
        p[k] = v
    params = p


    ####### Load model from checkpoint
    model = Moco_v2.load_from_checkpoint(
        checkpoint_path=params['checkpoint_path']
        )

    backbone = model.encoder_q
    backbone.fc = torch.nn.Identity()
    
    params['ssl_training'] = False
    params['jitter_lim'] = 0 #determines the maximum value for the offset in the jitter augmentation
    params['augmentations'] = 'jcrg' #adjust to whatever parameters you want, typically 'jcrg' or 'rrjc'
    
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
    
    representations = backbone(images)

    if params['gpu']:
        images, representations = images.detach(), representations.detach()
    images, representations = images.numpy(), representations.numpy()
    ###### smash down to 128 dimensions instead of 2048

    reps_128, reps_128_trans = dimensionality_reduction.umap_transform(representations, n_components=128, metric='cosine')
    
    np.save(f"/pscratch/sd/s/sgmoore1/ssl-legacysurvey/{params['rep_dir']}/{params['file_head']}_{0 :09d}_{params['ngals_tot']:09d}.npy", reps_128)















        