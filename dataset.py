import numpy as np
import matplotlib.pyplot as plt
import pickle
# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np

import glob 
import math

import torch # For building the networks 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam
import torch.optim as optim
import torch.nn.functional as F
from nystrom_attention import NystromAttention

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
from lifelines.utils import concordance_index
# import torchvision.transforms as transforms
from PIL import Image

# from TCGA_preprocess import TCGA_csv_process\
import sys

#Feature extraction
# from utils import perform_clustering, select_representative_images, extract_features
# import torchvision.models as models
import openslide as op
import cv2

import h5py 
# Define your custom dataset class
class SurvivalDataset(Dataset):
    def __init__(self, feat_paths, surv_data, clinical_data, num_patches = 10000, transform=None, mtlr = True):
        '''
        surv_data : ('TCGA-44-6777', 987.0, 1.0) [patient id, OS.time, OS]

        num_patch : int (>=50)
        '''
        self.feat_paths = feat_paths
        self.survival_data = surv_data
        self.clinical_data = clinical_data
        self.transform = transform
        
        self.mtlr = mtlr
        # self.cancer_classifier = cancer_classifier
    def __len__(self):
        return len(self.feat_paths)

    def __getitem__(self, idx):
        #max_num_patches = 144501
        desired_shape = (10000, 512)
       
        patches = torch.load(self.feat_paths[idx])
     
        if patches.shape[0] < desired_shape[0]:
            # If the tensor has more rows than the desired shape, pad it with zeros
            padding_needed = desired_shape[0] - patches.shape[0]
            patches = torch.nn.functional.pad(patches, (0, 0, 0, padding_needed))
        else:
            # Shuffle the indices
            shuffled_indices = torch.randperm(patches.size(0))
            
            # Use the shuffled indices to rearrange the rows of the tensor
            patches = patches[shuffled_indices][:desired_shape[0]]
                
        event = self.survival_data[idx][3]
        time = self.survival_data[idx][1]
        
        if self.mtlr:
            time_val = self.survival_data[idx][2]
            time = np.array(time, dtype=np.float32)
            time = torch.tensor(time, dtype=torch.float32)
            
         # Convert clinical_data to a numpy array with float type
        clinical_data = np.array(self.clinical_data[idx], dtype=np.float32)
        
        # Convert the numpy array to a PyTorch tensor
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        
        return patches, time, event, clinical_data, time_val
